from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from agent.llm_client import LLMClientError, LocalLLMClient
from agent.session_store import Message, Session, SessionStore

logger = logging.getLogger("agent.chat")

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, practical coding assistant. "
    "Prefer complete, runnable answers and say when you are unsure."
)

SUMMARY_PROMPT = (
    "Summarise the earlier part of this conversation for future reference. "
    "Keep decisions, file names, code identifiers and open questions. "
    "Write compact prose, at most 150 words."
)

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Fallback token estimate for clients without a real tokenizer.

    Roughly four characters per token, which underestimates dense code
    (measured at ~2.6 characters per token for the bundled model).
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class Context:
    messages: tuple[dict[str, str], ...]
    dropped: int
    tokens: int
    exact: bool = False


@dataclass(frozen=True)
class TurnTimings:
    prompt_ms: float
    summary_ms: float
    first_token_ms: float
    generation_ms: float
    python_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int | None
    tokens_per_second: float | None
    streamed: bool
    exact_tokens: bool = False

    def describe(self) -> str:
        speed = (
            f"{self.tokens_per_second:.1f}"
            if self.tokens_per_second is not None
            else "n/a"
        )
        completion = (
            "n/a"
            if self.completion_tokens is None
            else str(self.completion_tokens)
        )
        prompt_label = "prompt_tokens" if self.exact_tokens else "prompt_tokens~"
        return (
            f"prompt_ms={self.prompt_ms:.1f} "
            f"summary_ms={self.summary_ms:.1f} "
            f"first_token_ms={self.first_token_ms:.1f} "
            f"generation_ms={self.generation_ms:.1f} "
            f"python_ms={self.python_ms:.1f} "
            f"total_ms={self.total_ms:.1f} "
            f"{prompt_label}={self.prompt_tokens} "
            f"completion_tokens={completion} "
            f"tok_s={speed} stream={self.streamed}"
        )


@dataclass(frozen=True)
class TurnResult:
    text: str
    timings: TurnTimings
    dropped_messages: int


def build_context(
    history: Sequence[Message],
    system_prompt: str,
    summary: str,
    budget_tokens: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> Context:
    """Fit the newest contiguous turns into budget_tokens, plus the system part."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    system_text = system_prompt.strip()
    if summary:
        system_text = (
            f"{system_text}\n\nEarlier conversation summary:\n{summary.strip()}"
        ).strip()

    messages: list[dict[str, str]] = []
    used = 0
    if system_text:
        system_cost = count_tokens(system_text) + 4
        if system_cost > budget_tokens:
            raise ValueError(
                "system prompt and summary exceed the context budget"
            )
        messages.append({"role": "system", "content": system_text})
        used += system_cost

    kept: list[Message] = []
    for message in reversed(history):
        cost = count_tokens(message.content) + 4
        if used + cost > budget_tokens:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    messages.extend(
        {"role": item.role, "content": item.content} for item in kept
    )
    return Context(
        messages=tuple(messages),
        dropped=len(history) - len(kept),
        tokens=used,
    )


class ChatSession:
    """A persisted conversation that answers under a fixed context budget.

    Context layout per turn: system prompt, rolled-up summary of anything too
    old to fit, then as many recent messages as the budget allows.
    """

    def __init__(
        self,
        store: SessionStore,
        client: LocalLLMClient,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        context_tokens: int = 8192,
        max_tokens: int = 512,
        temperature: float = 0.0,
        summarize: bool = True,
        count_tokens: TokenCounter | None = None,
    ) -> None:
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_tokens >= context_tokens:
            raise ValueError(
                "max_tokens must be smaller than context_tokens"
            )

        self.store = store
        self.client = client
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.summarize = summarize

        # Prefer the client's exact tokenizer; the estimate is only a fallback.
        if count_tokens is None:
            candidate = getattr(client, "count_tokens", None)
            count_tokens = candidate if callable(candidate) else estimate_tokens
        self.count_tokens = count_tokens

        prompt_counter = getattr(client, "count_prompt_tokens", None)
        self._prompt_counter = prompt_counter if callable(prompt_counter) else None

        if session_id is None:
            self.session = store.create_session(
                system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
            )
        else:
            session = store.get_session(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            self.session = session

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def history(self, limit: int | None = None) -> list[Message]:
        return self.store.messages(self.session_id, limit=limit)

    def _budget(self) -> int:
        # Reserve space for the reply inside the same window.
        return self.context_tokens - self.max_tokens

    def _summarise(
        self, dropped: Sequence[Message]
    ) -> tuple[str, int, float]:
        """Fold the given turns into a summary.

        Returns ``(summary_text, watermark, elapsed_ms)`` where ``watermark`` is
        the highest message id the summary covers. Nothing is persisted here:
        the caller writes the summary only once the turn succeeds, and only the
        turns past the current watermark are ever passed in, so a turn is
        summarised exactly once.
        """
        if not dropped:
            return (
                self.session.summary,
                self.session.summarised_until,
                0.0,
            )

        lines = [f"{item.role}: {item.content}" for item in dropped]
        header_parts = [SUMMARY_PROMPT]
        if self.session.summary:
            header_parts.append(
                "Existing summary:\n" + self.session.summary
            )
        header = "\n\n".join(header_parts) + "\n\nConversation:\n"

        # The dropped transcript can be far larger than the model window, so
        # keep the most recent part of it and drop the oldest silently: older
        # material is already represented by the previous summary.
        reply_reserve = min(256, self.max_tokens)
        allowance = self._budget() - reply_reserve - 16
        transcript = "\n\n".join(lines)
        while transcript:
            try:
                fits = self.count_tokens(header + transcript) <= allowance
            except LLMClientError:
                fits = True
            if fits:
                break
            transcript = transcript[len(transcript) // 4:]

        started = time.perf_counter()
        try:
            response = self.client.chat(
                [{"role": "user", "content": header + transcript}],
                temperature=0.0,
                max_tokens=reply_reserve,
            )
        except LLMClientError as exc:
            # A failed summary must not lose the user's turn.
            logger.warning(
                "summary failed for session %s: %s", self.session_id, exc
            )
            return (
                self.session.summary,
                self.session.summarised_until,
                (time.perf_counter() - started) * 1000.0,
            )

        summary_ms = (time.perf_counter() - started) * 1000.0
        watermark = max(item.message_id for item in dropped)
        logger.info(
            "summarised %d messages for session %s in %.1f ms",
            len(dropped),
            self.session_id,
            summary_ms,
        )
        return response.text.strip(), watermark, summary_ms

    def _exact_prompt_tokens(
        self, messages: Sequence[dict[str, str]]
    ) -> int | None:
        if self._prompt_counter is None:
            return None
        try:
            return self._prompt_counter(list(messages))
        except LLMClientError as exc:
            logger.warning(
                "prompt counting failed for session %s: %s; using the estimate",
                self.session_id,
                exc,
            )
            return None

    def _trim_to_budget(self, context: Context) -> Context:
        """Drop whole turns until the rendered prompt fits the budget.

        The per-message estimate can be short by the chat-template overhead, so
        the exact rendered length is the authority whenever it is available.
        """
        if self._prompt_counter is None:
            return context

        budget = self._budget()
        messages = list(context.messages)
        dropped = context.dropped

        while True:
            exact = self._exact_prompt_tokens(messages)
            if exact is None:
                return context
            if exact <= budget:
                return Context(
                    messages=tuple(messages),
                    dropped=dropped,
                    tokens=exact,
                    exact=True,
                )

            # Never drop the system prompt or the newest message.
            removable = [
                index
                for index, message in enumerate(messages[:-1])
                if message["role"] != "system"
            ]
            if not removable:
                raise ValueError(
                    "message does not fit the context budget "
                    f"({exact} of {budget} tokens)"
                )
            messages.pop(removable[0])
            dropped += 1

    def send(
        self,
        text: str,
        *,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> TurnResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message must not be empty")

        started = time.perf_counter()
        assembling = time.perf_counter()

        # The prospective turn joins the history in memory, so its rendered
        # size is counted once and that exact context is what gets sent.
        pending = Message(
            message_id=0,
            session_id=self.session_id,
            role="user",
            content=text,
            created_at="",
            token_count=self.count_tokens(text),
        )
        history = self.history()
        candidate = list(history) + [pending]

        summary_text = self.session.summary
        watermark = self.session.summarised_until
        context = build_context(
            candidate,
            self.session.system_prompt,
            summary_text,
            self._budget(),
            self.count_tokens,
        )
        prompt_ms = (time.perf_counter() - assembling) * 1000.0

        summary_ms = 0.0
        if context.dropped and self.summarize:
            # Only turns that have never been folded into the summary; anything
            # at or below the watermark was already summarised on an earlier
            # turn and must not be summarised again.
            newly_dropped = [
                message
                for message in history[: context.dropped]
                if message.message_id > watermark
            ]
            if newly_dropped:
                summary_text, watermark, summary_ms = self._summarise(
                    newly_dropped
                )
                assembling = time.perf_counter()
                context = build_context(
                    candidate,
                    self.session.system_prompt,
                    summary_text,
                    self._budget(),
                    self.count_tokens,
                )
                prompt_ms += (time.perf_counter() - assembling) * 1000.0

        trimming = time.perf_counter()
        context = self._trim_to_budget(context)
        prompt_ms += (time.perf_counter() - trimming) * 1000.0

        # Nothing is written yet: the turn is accepted, and only then persisted,
        # if the measured prompt still contains the new message.
        newest = context.messages[-1] if context.messages else None
        if (
            newest is None
            or newest["role"] != "user"
            or newest["content"] != text
        ):
            message_tokens = pending.token_count or self.count_tokens(text)
            raise ValueError(
                "message does not fit the context budget "
                f"({message_tokens}-token message vs "
                f"{self._budget()} available)"
            )

        stored_user = self.store.append_message(
            self.session_id, "user", text, token_count=pending.token_count
        )

        try:
            if stream:
                stream_result = self.client.chat_stream(
                    list(context.messages),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                for delta in stream_result:
                    if on_token is not None:
                        on_token(delta)
                reply = stream_result.text
                first_token_ms = (
                    stream_result.first_token_seconds or 0.0
                ) * 1000.0
                generation_ms = max(
                    0.0,
                    (stream_result.latency_seconds or 0.0) * 1000.0
                    - first_token_ms,
                )
                completion_tokens = stream_result.completion_tokens
            else:
                response = self.client.chat(
                    list(context.messages),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                reply = response.text
                first_token_ms = response.latency_seconds * 1000.0
                generation_ms = 0.0
                completion_tokens = response.completion_tokens
        except BaseException:
            # Roll the turn back so an aborted request (error, timeout or a
            # Ctrl+C during the stream) never leaves an unanswered user
            # message behind. BaseException covers KeyboardInterrupt.
            self.store.delete_message(stored_user.message_id)
            raise

        # The turn only becomes durable once the model answered: a failed call
        # must not leave a user message with no reply behind it.
        if summary_text != self.session.summary or (
            watermark != self.session.summarised_until
        ):
            self.session = self.store.set_summary(
                self.session_id, summary_text, summarised_until=watermark
            )

        if completion_tokens is None:
            completion_tokens = self.count_tokens(reply)

        self.store.append_message(
            self.session_id,
            "assistant",
            reply,
            token_count=completion_tokens,
        )

        total_ms = (time.perf_counter() - started) * 1000.0
        llm_ms = summary_ms + first_token_ms + generation_ms
        tokens_per_second = None
        if generation_ms > 0 and completion_tokens:
            tokens_per_second = completion_tokens / (generation_ms / 1000.0)

        timings = TurnTimings(
            prompt_ms=prompt_ms,
            summary_ms=summary_ms,
            first_token_ms=first_token_ms,
            generation_ms=generation_ms,
            python_ms=max(0.0, total_ms - llm_ms),
            total_ms=total_ms,
            prompt_tokens=context.tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            streamed=stream,
            exact_tokens=context.exact,
        )
        logger.info(
            "turn session=%s messages=%d dropped=%d %s",
            self.session_id,
            len(context.messages),
            context.dropped,
            timings.describe(),
        )

        return TurnResult(
            text=reply,
            timings=timings,
            dropped_messages=context.dropped,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent.chat",
        description="Chat with a local model through ChatSession.",
    )
    parser.add_argument(
        "--db", default="state/chat.sqlite3",
        help="session database (default: state/chat.sqlite3)",
    )
    parser.add_argument(
        "--session", default=None, help="resume an existing session id"
    )
    parser.add_argument(
        "--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt"
    )
    parser.add_argument(
        "--context-tokens", type=int, default=8192,
        help="context window; must match llama-server --ctx-size (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="tokens reserved for each reply (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="sampling temperature"
    )
    parser.add_argument("--base-url", default=None, help="inference server URL")
    parser.add_argument("--model", default=None, help="model id to request")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="request timeout in seconds"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="buffer replies instead of streaming"
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_sessions",
        help="list recent sessions and exit",
    )
    return parser.parse_args(argv)


def _print_sessions(store: SessionStore, limit: int = 20) -> None:
    sessions = store.list_sessions(limit=limit)
    if not sessions:
        print("no sessions yet")
        return

    for session in sessions:
        opening = ""
        messages = store.messages(session.session_id, limit=1)
        if messages:
            opening = messages[0].content.splitlines()[0][:60]
        print(f"{session.session_id}  {session.updated_at}  {opening}")


def _print_help() -> None:
    print(
        "/help     show commands\n"
        "/new      start a new session\n"
        "/session  print the current session id\n"
        "/exit     quit (Ctrl+D also works)"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: LocalLLMClient | None = None,
) -> int:
    args = _parse_args(argv)

    store = SessionStore(args.db)
    if args.list_sessions:
        _print_sessions(store)
        return 0

    if client is None:
        client = LocalLLMClient(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
        )

    def new_session(session_id: str | None = None) -> ChatSession:
        return ChatSession(
            store,
            client,
            session_id=session_id,
            system_prompt=args.system,
            context_tokens=args.context_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    try:
        chat = new_session(args.session)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"session {chat.session_id} | db {args.db} | "
        f"context {chat.context_tokens} (reply {chat.max_tokens})"
    )
    print("type /help for commands, /exit to quit\n")

    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/help":
            _print_help()
            continue
        if text == "/session":
            print(chat.session_id)
            continue
        if text == "/new":
            chat = new_session()
            print(f"new session {chat.session_id}")
            continue

        print("agent> ", end="", flush=True)
        try:
            if args.no_stream:
                result = chat.send(text, stream=False)
                print(result.text)
            else:
                result = chat.send(
                    text,
                    on_token=lambda piece: print(piece, end="", flush=True),
                )
                print()
        except KeyboardInterrupt:
            # The turn was rolled back in ChatSession; keep the session alive.
            print("\n[cancelled]")
            continue
        except (LLMClientError, ValueError) as exc:
            print(f"\nerror: {exc}")
            continue

        if result.dropped_messages:
            print(f"[dropped {result.dropped_messages} old messages]")
        print(f"[{result.timings.describe()}]")


if __name__ == "__main__":
    raise SystemExit(main())
