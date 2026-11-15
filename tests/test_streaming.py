import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.llm_client import (
    LLMHTTPError,
    LLMResponseError,
    LocalLLMClient,
)


class SSEHandler(BaseHTTPRequestHandler):
    mode = "stream"
    last_request = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "mock-model"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_event(self, payload):
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        SSEHandler.last_request = json.loads(
            self.rfile.read(length).decode("utf-8")
        )

        if SSEHandler.mode == "http_error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        self.wfile.write(b": keep-alive\n\n")
        self.send_event(json.dumps({
            "model": "mock-model",
            "choices": [{"delta": {"role": "assistant"}}],
        }))

        if SSEHandler.mode == "malformed":
            self.send_event("{not json")
            return

        for index, piece in enumerate(("Hello", ", ", "world")):
            choice = {"delta": {"content": piece}}
            if SSEHandler.last_request.get("logprobs"):
                choice["logprobs"] = {
                    "content": [
                        {"token": piece, "logprob": (-0.1, -0.2, -0.3)[index]}
                    ]
                }
            self.send_event(json.dumps({"choices": [choice]}))

        self.send_event(json.dumps({
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        }))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        if cls.thread.is_alive():
            raise RuntimeError("Mock HTTP server thread did not stop")

    def setUp(self):
        SSEHandler.mode = "stream"
        SSEHandler.last_request = None
        self.client = LocalLLMClient(
            base_url=self.base_url, timeout_seconds=2.0
        )

    def test_deltas_accumulate_into_full_text(self):
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}]
        )

        deltas = list(stream)

        self.assertEqual(deltas, ["Hello", ", ", "world"])
        self.assertEqual(stream.text, "Hello, world")
        self.assertEqual(stream.model, "mock-model")
        self.assertEqual(stream.prompt_tokens, 11)
        self.assertEqual(stream.completion_tokens, 3)
        self.assertEqual(stream.total_tokens, 14)
        self.assertIsNotNone(stream.first_token_seconds)
        self.assertIsNotNone(stream.latency_seconds)
        self.assertGreaterEqual(
            stream.latency_seconds, stream.first_token_seconds
        )
        self.assertTrue(SSEHandler.last_request["stream"])

    def test_stream_collects_token_logprobs_when_requested(self):
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}], logprobs=True
        )

        list(stream)

        self.assertIs(SSEHandler.last_request["logprobs"], True)
        self.assertEqual(stream.token_logprobs, (-0.1, -0.2, -0.3))

    def test_stream_without_logprobs_stays_empty(self):
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}]
        )

        list(stream)

        self.assertNotIn("logprobs", SSEHandler.last_request)
        self.assertEqual(stream.token_logprobs, ())

    def test_stream_can_only_be_iterated_once(self):
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}]
        )
        list(stream)

        with self.assertRaises(RuntimeError):
            list(stream)

    def test_malformed_chunk_is_rejected(self):
        SSEHandler.mode = "malformed"
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}]
        )

        with self.assertRaises(LLMResponseError):
            list(stream)

    def test_http_error_is_distinct(self):
        SSEHandler.mode = "http_error"
        stream = self.client.chat_stream(
            [{"role": "user", "content": "hi"}]
        )

        with self.assertRaises(LLMHTTPError) as caught:
            list(stream)
        self.assertEqual(caught.exception.status_code, 500)

    def test_invalid_arguments_rejected_before_request(self):
        with self.assertRaises(ValueError):
            self.client.chat_stream([])
        with self.assertRaises(ValueError):
            self.client.chat_stream(
                [{"role": "user", "content": "hi"}], max_tokens=0
            )


if __name__ == "__main__":
    unittest.main()
