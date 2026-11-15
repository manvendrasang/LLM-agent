import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.chat import main
from agent.llm_client import LLMResponse
from agent.session_store import SessionStore


class FakeStream:
    def __init__(self, text):
        self._text = text
        self.text = ""
        self.token_logprobs = ()
        self.model = "fake-model"
        self.prompt_tokens = 10
        self.completion_tokens = 4
        self.total_tokens = 14
        self.first_token_seconds = 0.01
        self.latency_seconds = 0.04

    def __iter__(self):
        accumulated = ""
        for piece in self._text:
            accumulated += piece
            self.text = accumulated
            yield piece


class CliClient:
    def __init__(self, reply="hello from the model"):
        self.reply = reply
        self.stream_calls = 0
        self.buffered_calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=256):
        self.buffered_calls += 1
        return LLMResponse(
            text=self.reply,
            model="fake-model",
            prompt_tokens=5,
            completion_tokens=4,
            total_tokens=9,
            latency_seconds=0.02,
        )

    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.stream_calls += 1
        return FakeStream(self.reply)


class InterruptingStream(FakeStream):
    def __iter__(self):
        for piece in "partial reply":
            self.text += piece
            yield piece
        raise KeyboardInterrupt


class InterruptingCliClient(CliClient):
    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.stream_calls += 1
        return InterruptingStream("")


class CountingCliClient(CliClient):
    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def count_prompt_tokens(self, messages):
        return sum(len(message["content"]) // 4 + 40 for message in messages)


def db_path(temp_dir):
    return str(Path(temp_dir.name) / "chat.sqlite3")


class ChatCliTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.db = db_path(temp_dir)

    def run_cli(self, argv, inputs, client):
        out, err = io.StringIO(), io.StringIO()
        with patch("builtins.input", side_effect=inputs):
            with redirect_stdout(out), redirect_stderr(err):
                code = main(argv + ["--db", self.db], client=client)
        return code, out.getvalue(), err.getvalue()

    def test_streams_reply_and_prints_timing_line(self):
        client = CliClient(reply="the answer")
        code, out, _ = self.run_cli([], ["how do I reverse a list?", "/exit"], client)

        self.assertEqual(code, 0)
        self.assertIn("agent> the answer", out.replace("\n", ""))
        self.assertIn("prompt_ms=", out)
        self.assertIn("first_token_ms=", out)
        self.assertIn("stream=True", out)
        self.assertEqual(client.stream_calls, 1)

        session = SessionStore(self.db).list_sessions()[0]
        self.assertEqual(len(SessionStore(self.db).messages(session.session_id)), 2)

    def test_missing_repo_index_exits_two(self):
        client = CliClient(reply="never used")

        code, _, err = self.run_cli(
            ["--repo-index", self.db + ".missing"], [], client
        )

        self.assertEqual(code, 2)
        self.assertIn("index not found", err)
        self.assertEqual(client.stream_calls, 0)

    def test_no_stream_flag_uses_buffered_request(self):
        client = CliClient(reply="buffered")
        code, out, _ = self.run_cli(
            ["--no-stream"], ["hello", "/exit"], client
        )

        self.assertEqual(code, 0)
        self.assertIn("buffered", out)
        self.assertIn("stream=False", out)
        self.assertEqual(client.buffered_calls, 1)

    def test_oversized_message_is_refused_and_not_stored(self):
        client = CountingCliClient()
        code, out, _ = self.run_cli(
            ["--context-tokens", "100", "--max-tokens", "20"],
            ["x" * 4000, "/exit"],
            client,
        )

        self.assertEqual(code, 0)
        self.assertIn("error:", out)
        self.assertIn("context budget", out)

        session = SessionStore(self.db).list_sessions()[0]
        self.assertEqual(SessionStore(self.db).messages(session.session_id), [])
        self.assertEqual(client.stream_calls, 0)

    def test_session_command_and_new_session(self):
        client = CliClient()
        code, out, _ = self.run_cli(
            ["--session", "resume-me-not"], ["/exit"], client
        )
        self.assertEqual(code, 2)

        code, out, _ = self.run_cli([], ["/session", "/new", "/exit"], client)
        self.assertEqual(code, 0)
        self.assertIn("session ", out)
        self.assertIn("new session ", out)

    def test_ctrl_c_during_a_reply_cancels_the_turn_and_keeps_running(self):
        client = InterruptingCliClient()
        code, out, _ = self.run_cli([], ["hello", "/exit"], client)

        self.assertEqual(code, 0)
        self.assertIn("[cancelled]", out)
        self.assertIn("partial reply", out)
        session = SessionStore(self.db).list_sessions()[0]
        self.assertEqual(SessionStore(self.db).messages(session.session_id), [])

    def test_list_prints_existing_sessions(self):
        store = SessionStore(self.db)
        session = store.create_session("system prompt")
        store.append_message(session.session_id, "user", "first words here")

        client = CliClient()
        code, out, _ = self.run_cli(["--list"], [], client)

        self.assertEqual(code, 0)
        self.assertIn(session.session_id, out)
        self.assertIn("first words here", out)
        self.assertEqual(client.stream_calls, 0)


if __name__ == "__main__":
    unittest.main()
