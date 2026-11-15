from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agent.chat import DEFAULT_SYSTEM_PROMPT, ChatSession
from agent.llm_client import LLMClientError, LocalLLMClient
from agent.session_store import SessionStore

DEFAULT_CASES_PATH = "evals/cases.json"

# Phrases that count as the model declining to answer. Kept deliberately broad
# and lowercase; a refusal is a behaviour to measure, not a precise signal.
REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i can not",
    "i won't",
    "i will not",
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "cannot know",
    "can't know",
    "cannot help",
    "can't help",
    "cannot provide",
    "can't provide",
    "unable to",
    "not able to",
    "no access to",
    "no way to",
    "not publicly available",
    "cannot verify",
    "can't verify",
)

CASE_FIELDS = {
    "id",
    "prompt",
    "expect_contains",
    "expect_any",
    "expect_absent",
    "expect_refusal",
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expect_contains: tuple[str, ...] = ()
    expect_any: tuple[str, ...] = ()
    expect_absent: tuple[str, ...] = ()
    expect_refusal: bool | None = None


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    reply: str
    passed: bool
    failures: tuple[str, ...]
    refused: bool
    total_ms: float
    first_token_ms: float


@dataclass(frozen=True)
class EvalSummary:
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def refusal_rate(self) -> float:
        if not self.results:
            return 0.0
        refused = sum(1 for result in self.results if result.refused)
        return refused / len(self.results)

    @property
    def latency_p50(self) -> float:
        return percentile([result.total_ms for result in self.results], 0.5)

    @property
    def latency_p95(self) -> float:
        return percentile([result.total_ms for result in self.results], 0.95)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def is_refusal(reply: str) -> bool:
    lowered = reply.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _string_tuple(raw: dict, key: str, case_id: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"case {case_id!r}: {key} must be a list of strings")
    return tuple(value)


def _parse_case(raw: object, seen: set[str]) -> EvalCase:
    if not isinstance(raw, dict):
        raise ValueError("each case must be a JSON object")

    unknown = set(raw) - CASE_FIELDS
    if unknown:
        raise ValueError(f"unknown case fields: {sorted(unknown)}")

    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("each case needs a non-empty 'id'")
    if case_id in seen:
        raise ValueError(f"duplicate case id: {case_id!r}")
    seen.add(case_id)

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"case {case_id!r}: prompt must be a non-empty string")

    expect_refusal = raw.get("expect_refusal")
    if expect_refusal is not None and not isinstance(expect_refusal, bool):
        raise ValueError(
            f"case {case_id!r}: expect_refusal must be true, false or omitted"
        )

    return EvalCase(
        id=case_id,
        prompt=prompt,
        expect_contains=_string_tuple(raw, "expect_contains", case_id),
        expect_any=_string_tuple(raw, "expect_any", case_id),
        expect_absent=_string_tuple(raw, "expect_absent", case_id),
        expect_refusal=expect_refusal,
    )


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError("case file must contain a non-empty list of cases")

    seen: set[str] = set()
    return [_parse_case(item, seen) for item in raw]


def evaluate(case: EvalCase, reply: str) -> tuple[str, ...]:
    lowered = reply.lower()
    failures: list[str] = []

    for needle in case.expect_contains:
        if needle.lower() not in lowered:
            failures.append(f"missing {needle!r}")

    if case.expect_any and not any(
        needle.lower() in lowered for needle in case.expect_any
    ):
        failures.append("none of " + ", ".join(repr(n) for n in case.expect_any))

    for needle in case.expect_absent:
        if needle.lower() in lowered:
            failures.append(f"contains {needle!r}")

    refused = is_refusal(reply)
    if case.expect_refusal is True and not refused:
        failures.append("expected a refusal")
    elif case.expect_refusal is False and refused:
        failures.append("unexpected refusal")

    return tuple(failures)


def run_case(
    store: SessionStore,
    client: LocalLLMClient,
    case: EvalCase,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context_tokens: int = 8192,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> CaseResult:
    try:
        chat = ChatSession(
            store,
            client,
            system_prompt=system_prompt,
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = chat.send(case.prompt)
    except LLMClientError as exc:
        return CaseResult(
            case=case,
            reply="",
            passed=False,
            failures=(f"request failed: {type(exc).__name__}: {exc}",),
            refused=False,
            total_ms=0.0,
            first_token_ms=0.0,
        )

    failures = evaluate(case, result.text)
    return CaseResult(
        case=case,
        reply=result.text,
        passed=not failures,
        failures=failures,
        refused=is_refusal(result.text),
        total_ms=result.timings.total_ms,
        first_token_ms=result.timings.first_token_ms,
    )


def run_evals(
    cases: Sequence[EvalCase],
    store: SessionStore,
    client: LocalLLMClient,
    **kwargs,
) -> EvalSummary:
    # Each case gets its own session so one case cannot leak context into the
    # next one.
    results = [run_case(store, client, case, **kwargs) for case in cases]
    return EvalSummary(results=tuple(results))


def format_summary(summary: EvalSummary) -> str:
    lines: list[str] = []
    for result in summary.results:
        mark = "PASS" if result.passed else "FAIL"
        first_line = result.reply.strip().splitlines()
        preview = first_line[0][:60] if first_line else ""
        lines.append(
            f"{mark}  {result.case.id:<24} {result.total_ms:7.0f} ms  {preview}"
        )
        for failure in result.failures:
            lines.append(f"      - {failure}")

    lines.append("")
    lines.append(
        f"cases {len(summary.results)} | passed {summary.passed} | "
        f"failed {summary.failed}"
    )
    lines.append(
        f"refusal rate {summary.refusal_rate:.0%} | "
        f"p50 {summary.latency_p50:.0f} ms | p95 {summary.latency_p95:.0f} ms"
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent.evals",
        description="Run chat eval cases against the local model.",
    )
    parser.add_argument(
        "--cases", default=DEFAULT_CASES_PATH,
        help=f"JSON case file (default: {DEFAULT_CASES_PATH})",
    )
    parser.add_argument(
        "--db", default="state/evals.sqlite3",
        help="session database for the runs (default: state/evals.sqlite3)",
    )
    parser.add_argument(
        "--only", default=None, metavar="ID",
        help="run a single case by id",
    )
    parser.add_argument(
        "--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt"
    )
    parser.add_argument(
        "--context-tokens", type=int, default=8192,
        help="context window; must match llama-server --ctx-size",
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
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: LocalLLMClient | None = None,
) -> int:
    args = _parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.only is not None:
        cases = [case for case in cases if case.id == args.only]
        if not cases:
            print(f"error: no case with id {args.only!r}", file=sys.stderr)
            return 2

    if client is None:
        client = LocalLLMClient(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
        )

    store = SessionStore(args.db)
    summary = run_evals(
        cases,
        store,
        client,
        system_prompt=args.system,
        context_tokens=args.context_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(format_summary(summary))
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
