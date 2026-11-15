import tempfile
import unittest
from pathlib import Path

from agent.chat import (
    ChatSession,
    build_context,
    estimate_tokens,
)
from agent.llm_client import LLMConnectionError, LLMResponse
from agent.session_store import SessionStore

SUMMARY_MARKER = "Summarise the earlier part"


class FakeStream:
    def __init__(self, text, first=0.02, latency=0.06, completion=9):
        self._text = text
        self.text = ""
        self.model = "fake-model"
        self.prompt_tokens = 100
        self.completion_tokens = completion
        self.total_tokens = None
        self.first_token_seconds = first
        self.latency_seconds = latency

    def __iter__(self):
        accumulated = ""
        for piece in self._text:
            accumulated += piece
            self.text = accumulated
            yield piece


class FakeClient:
    def __init__(self, reply="ok", summary="rolled up earlier turns",
                 fail_summary=False):
        self.reply = reply
        self.summary = summary
        self.fail_summary = fail_summary
        self.stream_calls = []
        self.buffered_calls = []
        self.summary_calls = []

    def chat(self, messages, *, temperature=0.0, max_tokens=256):
        if SUMMARY_MARKER in messages[-1]["content"]:
            self.summary_calls.append(messages[-1]["content"])
            if self.fail_summary:
                raise LLMConnectionError("summary server went away")
            return LLMResponse(
                text=self.summary,
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_seconds=0.01,
            )

        self.buffered_calls.append(list(messages))
        return LLMResponse(
            text=self.reply,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.05,
        )

    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.stream_calls.append(list(messages))
        return FakeStream(self.reply)


class FailingStream:
    def __init__(self, text, fail_after):
        self._text = text
        self.text = ""
        self.model = "fake-model"
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.first_token_seconds = 0.01
        self.latency_seconds = 0.02
        self.fail_after = fail_after

    def __iter__(self):
        accumulated = ""
        for index, piece in enumerate(self._text):
            if index == self.fail_after:
                raise LLMConnectionError("stream broke mid answer")
            accumulated += piece
            self.text = accumulated
            yield piece


class FailingStreamClient(FakeClient):
    def __init__(self, *args, fail_after=3, **kwargs):
        self.fail_after = fail_after
        super().__init__(*args, **kwargs)

    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.stream_calls.append(list(messages))
        return FailingStream("partial reply", self.fail_after)


class InterruptingStream:
    def __init__(self, text="partial reply"):
        self._text = text
        self.text = ""
        self.model = "fake-model"
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.first_token_seconds = 0.01
        self.latency_seconds = 0.02

    def __iter__(self):
        for piece in self._text:
            self.text += piece
            yield piece
        raise KeyboardInterrupt


class InterruptingStreamClient(FakeClient):
    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.stream_calls.append(list(messages))
        return InterruptingStream()


class FailingChatClient(FakeClient):
    def chat(self, messages, *, temperature=0.0, max_tokens=256):
        raise LLMConnectionError("server went away")


class ExactCountingClient(FakeClient):
    """Counts tokens exactly, and deliberately counts the template overhead."""

    def __init__(self, *args, **kwargs):
        self.prompt_count_calls = 0
        super().__init__(*args, **kwargs)

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def count_prompt_tokens(self, messages):
        self.prompt_count_calls += 1
        return sum(len(message["content"]) // 4 + 40 for message in messages)


class ChatSessionTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.db_path = Path(temp_dir.name) / "chat.sqlite3"
        self.store = SessionStore(self.db_path)
        self.client = FakeClient(reply="the answer")

    def make_session(self, **kwargs):
        return ChatSession(self.store, self.client, **kwargs)

    def test_turn_persists_user_and_assistant_messages(self):
        chat = self.make_session(system_prompt="be terse")

        result = chat.send("how do I reverse a list?")

        self.assertEqual(result.text, "the answer")
        history = chat.history()
        self.assertEqual([item.role for item in history], ["user", "assistant"])
        self.assertEqual(history[0].content, "how do I reverse a list?")
        self.assertEqual(history[1].content, "the answer")

    def test_system_prompt_is_first_message(self):
        chat = self.make_session(system_prompt="be terse")

        chat.send("hello")

        sent = self.client.stream_calls[0]
        self.assertEqual(sent[0], {"role": "system", "content": "be terse"})
        self.assertEqual(sent[-1], {"role": "user", "content": "hello"})

    def test_budget_drops_oldest_messages(self):
        chat = self.make_session(
            system_prompt="sys", context_tokens=220, max_tokens=20
        )
        for index in range(6):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 400
            )

        result = chat.send("newest question")

        self.assertGreater(result.dropped_messages, 0)
        sent = self.client.stream_calls[0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[-1]["content"], "newest question")
        self.assertNotIn("old 0", sent[1]["content"])
        self.assertLessEqual(result.timings.prompt_tokens, 200)

    def test_overflow_creates_and_persists_summary(self):
        chat = self.make_session(
            system_prompt="sys", context_tokens=220, max_tokens=20
        )
        for index in range(4):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 400
            )

        result = chat.send("new question")

        self.assertTrue(self.client.summary_calls)
        self.assertIn(SUMMARY_MARKER, self.client.summary_calls[0])
        self.assertGreaterEqual(result.timings.summary_ms, 0.0)
        self.assertIn("summary_ms=", result.timings.describe())
        recovered = self.store.get_session(chat.session_id)
        self.assertEqual(recovered.summary, "rolled up earlier turns")
        self.assertGreater(result.dropped_messages, 0)
        system_message = self.client.stream_calls[0][0]
        self.assertIn("rolled up earlier turns", system_message["content"])

    def test_summary_failure_still_answers(self):
        self.client.fail_summary = True
        chat = self.make_session(
            system_prompt="sys", context_tokens=220, max_tokens=20
        )
        for index in range(4):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 400
            )

        with self.assertLogs("agent.chat", level="WARNING"):
            result = chat.send("does it still work?")

        self.assertEqual(result.text, "the answer")
        self.assertEqual(self.store.get_session(chat.session_id).summary, "")

    def test_timings_are_returned_and_logged(self):
        chat = self.make_session()

        with self.assertLogs("agent.chat", level="INFO") as captured:
            result = chat.send("measure me")

        timings = result.timings
        self.assertGreaterEqual(timings.prompt_ms, 0.0)
        self.assertGreater(timings.first_token_ms, 0.0)
        self.assertGreaterEqual(timings.generation_ms, 0.0)
        self.assertGreaterEqual(timings.python_ms, 0.0)
        self.assertGreater(timings.total_ms, 0.0)
        self.assertGreater(timings.prompt_tokens, 0)
        self.assertEqual(timings.completion_tokens, 9)
        self.assertTrue(timings.streamed)
        self.assertIn("prompt_ms=", timings.describe())
        self.assertEqual(timings.summary_ms, 0.0)
        self.assertIn("first_token_ms=", timings.describe())
        self.assertIn("python_ms=", timings.describe())
        self.assertIn("stream=True", timings.describe())
        self.assertTrue(
            any("prompt_ms=" in line for line in captured.output)
        )

    def test_on_token_receives_each_streamed_piece(self):
        chat = self.make_session()
        pieces = []

        result = chat.send("stream it", on_token=pieces.append)

        self.assertEqual("".join(pieces), result.text)
        self.assertGreater(len(pieces), 1)

    def test_rendered_check_rejects_before_any_message_is_stored(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=200, max_tokens=20,
        )

        # 500 chars is only ~125 tokens, which the estimate would accept; the
        # rendered prompt with template overhead is over the 180 token budget.
        with self.assertRaises(ValueError):
            chat.send("x" * 500)

        self.assertEqual(chat.history(), [])
        self.assertFalse(client.stream_calls)

    def test_non_streaming_turn_uses_chat(self):
        chat = self.make_session()

        result = chat.send("no streaming please", stream=False)

        self.assertEqual(result.text, "the answer")
        self.assertFalse(result.timings.streamed)
        self.assertTrue(self.client.buffered_calls)
        self.assertFalse(self.client.stream_calls)

    def test_history_continues_after_reopen(self):
        chat = self.make_session()
        chat.send("first turn")

        reopened = ChatSession(
            SessionStore(self.db_path),
            self.client,
            session_id=chat.session_id,
        )
        reopened.send("second turn")

        sent = self.client.stream_calls[-1]
        contents = [message["content"] for message in sent]
        self.assertIn("first turn", contents)
        self.assertEqual(len(reopened.history()), 4)

    def test_empty_message_is_rejected(self):
        chat = self.make_session()
        for text in ("", "   ", "\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    chat.send(text)

    def test_invalid_budget_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_session(context_tokens=100, max_tokens=100)
        with self.assertRaises(ValueError):
            self.make_session(context_tokens=500, max_tokens=0)

    def test_unknown_session_id_raises(self):
        with self.assertRaises(KeyError):
            self.make_session(session_id="missing")

    def test_build_context_rejects_oversized_system_prompt(self):
        with self.assertRaises(ValueError):
            build_context([], "x" * 4000, "", 100)

    def test_estimate_fallback_is_used_without_a_server(self):
        chat = self.make_session()
        self.assertIs(chat.count_tokens, estimate_tokens)
        self.assertIsNone(chat._prompt_counter)

    def test_exact_count_trims_messages_the_estimate_would_keep(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=400, max_tokens=40,
        )
        for index in range(3):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 400
            )

        result = chat.send("question?")

        sent = client.stream_calls[0]
        # 3 system-or-history messages survived, 2 were dropped by the exact
        # count even though the four-chars-per-token estimate fit them all.
        self.assertEqual(len(sent), 3)
        self.assertEqual(result.dropped_messages, 2)
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[-1]["content"], "question?")
        # One count for the accepted prompt plus one per dropped message. This
        # is checked before the helper below, which counts a prompt itself.
        self.assertEqual(client.prompt_count_calls, 3)
        self.assertLessEqual(
            client.count_prompt_tokens(sent), chat.context_tokens - 40
        )

    def test_summary_prompt_itself_fits_the_budget(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=400, max_tokens=40,
        )
        for index in range(5):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 2000
            )

        chat.send("next question")

        self.assertTrue(client.summary_calls)
        summary_prompt = client.summary_calls[0]
        allowance = 360 - 40 - 16
        self.assertLessEqual(client.count_tokens(summary_prompt), allowance)

    def test_dropped_turns_are_summarised_once_across_turns(self):
        chat = self.make_session(
            system_prompt="sys", context_tokens=700, max_tokens=60
        )
        for index in range(5):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 800
            )

        first = chat.send("first question")
        self.assertGreater(first.dropped_messages, 0)
        self.assertEqual(len(self.client.summary_calls), 1)
        watermark = self.store.get_session(chat.session_id).summarised_until
        self.assertGreater(watermark, 0)

        chat.send("second question")

        covered = "\n".join(self.client.summary_calls)
        self.assertEqual(covered.count("old 0 "), 1)
        self.assertEqual(covered.count("old 1 "), 1)
        # Anything summarised on the first turn must not be summarised again.
        self.assertNotIn("old 0", self.client.summary_calls[-1])
        self.assertNotIn("old 1", self.client.summary_calls[-1])

    def test_turns_at_or_below_the_watermark_are_not_summarised_again(self):
        chat = self.make_session(
            system_prompt="sys", context_tokens=700, max_tokens=60
        )
        seeded = [
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 800
            )
            for index in range(5)
        ]
        self.store.set_summary(
            chat.session_id,
            "earlier context",
            summarised_until=seeded[-1].message_id,
        )
        chat.session = self.store.get_session(chat.session_id)

        result = chat.send("brand new question")

        self.assertGreater(result.dropped_messages, 0)
        self.assertEqual(self.client.summary_calls, [])

    def test_failed_stream_rolls_back_the_user_message(self):
        client = FailingStreamClient()
        chat = ChatSession(self.store, client, system_prompt="sys")

        with self.assertRaises(LLMConnectionError):
            chat.send("this turn will break")

        self.assertEqual(chat.history(), [])
        self.assertEqual(self.store.messages(chat.session_id), [])

    def test_ctrl_c_during_a_stream_rolls_back_the_user_message(self):
        client = InterruptingStreamClient()
        chat = ChatSession(self.store, client, system_prompt="sys")
        streamed = []

        with self.assertRaises(KeyboardInterrupt):
            chat.send("interrupt me", on_token=streamed.append)

        self.assertEqual("".join(streamed), "partial reply")
        self.assertEqual(chat.history(), [])
        self.assertEqual(len(self.store.messages(chat.session_id)), 0)

    def test_failed_buffered_request_rolls_back_the_user_message(self):
        client = FailingChatClient()
        chat = ChatSession(self.store, client, system_prompt="sys")

        with self.assertRaises(LLMConnectionError):
            chat.send("fails before answering", stream=False)

        self.assertEqual(chat.history(), [])

    def test_failure_keeps_earlier_turns_intact(self):
        client = FailingStreamClient()
        chat = ChatSession(self.store, client, system_prompt="sys")
        self.store.append_message(chat.session_id, "user", "earlier question")
        self.store.append_message(chat.session_id, "assistant", "earlier answer")

        with self.assertRaises(LLMConnectionError):
            chat.send("fails now")

        self.assertEqual(
            [message.role for message in chat.history()],
            ["user", "assistant"],
        )

    def test_a_fitting_turn_counts_the_rendered_prompt_once(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=8192, max_tokens=64,
        )

        chat.send("a normal question")

        self.assertEqual(client.prompt_count_calls, 1)

    def test_rejected_turn_persists_neither_message_nor_summary(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=200, max_tokens=20,
        )
        for index in range(6):
            self.store.append_message(
                chat.session_id, "user", f"old {index} " + "x" * 2000
            )

        with self.assertRaises(ValueError):
            chat.send("y" * 4000)

        self.assertTrue(client.summary_calls)
        self.assertEqual(self.store.get_session(chat.session_id).summary, "")
        self.assertEqual(len(chat.history()), 6)

    def test_estimate_only_client_rejects_a_message_that_cannot_fit(self):
        chat = self.make_session(
            system_prompt="sys", context_tokens=100, max_tokens=20
        )

        with self.assertRaises(ValueError):
            chat.send("x" * 4000)

        self.assertEqual(chat.history(), [])
        self.assertFalse(self.client.stream_calls)

    def test_exact_count_rejects_a_message_larger_than_the_budget(self):
        client = ExactCountingClient(reply="answer")
        chat = ChatSession(
            self.store, client, system_prompt="sys",
            context_tokens=100, max_tokens=20,
        )

        with self.assertRaises(ValueError):
            chat.send("x" * 4000)

        self.assertEqual(chat.history(), [])
        self.assertFalse(client.stream_calls)

    def test_estimate_tokens_is_monotonic(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertLess(estimate_tokens("short"), estimate_tokens("x" * 100))


if __name__ == "__main__":
    unittest.main()
