import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.llm_client import (
    LLMConnectionError,
    LLMHTTPError,
    LLMResponseError,
    LLMTimeoutError,
    LocalLLMClient,
)


class MockHandler(BaseHTTPRequestHandler):
    mode = "success"
    last_request = None

    def log_message(self, format, *args):
        pass

    def respond(self, status, body):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()

        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def do_GET(self):
        if self.path == "/v1/models":
            self.respond(200, {
                "data": [{"id": "mock-model"}]
            })
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        MockHandler.last_request = json.loads(
            self.rfile.read(length).decode("utf-8")
        )

        if MockHandler.mode == "http_error":
            self.respond(500, {"error": "mock failure"})
        elif MockHandler.mode == "malformed":
            body = b"not-json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif MockHandler.mode == "bad_shape":
            self.respond(200, {"choices": []})
        elif MockHandler.mode == "delay":
            time.sleep(0.2)
            self.respond(200, {
                "model": "mock-model",
                "choices": [{
                    "message": {"content": "late"}
                }],
                "usage": {},
            })
        else:
            self.respond(200, {
                "model": "mock-model",
                "choices": [{
                    "message": {"content": "LOCAL_MODEL_OK"}
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            })


class LLMClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
        )
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
        MockHandler.mode = "success"
        MockHandler.last_request = None
        self.client = LocalLLMClient(
            base_url=self.base_url,
            timeout_seconds=1.0,
        )

    def test_success_discovers_model_and_returns_usage(self):
        result = self.client.chat([
            {"role": "user", "content": "test"}
        ])

        self.assertEqual(result.text, "LOCAL_MODEL_OK")
        self.assertEqual(result.model, "mock-model")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.completion_tokens, 4)
        self.assertEqual(result.total_tokens, 14)
        self.assertGreaterEqual(result.latency_seconds, 0)
        self.assertEqual(MockHandler.last_request["model"], "mock-model")

    def test_http_error_is_distinct(self):
        MockHandler.mode = "http_error"
        with self.assertRaises(LLMHTTPError) as caught:
            self.client.chat([{"role": "user", "content": "test"}])
        self.assertEqual(caught.exception.status_code, 500)

    def test_malformed_json_is_rejected(self):
        MockHandler.mode = "malformed"
        with self.assertRaises(LLMResponseError):
            self.client.chat([{"role": "user", "content": "test"}])

    def test_unexpected_response_shape_is_rejected(self):
        MockHandler.mode = "bad_shape"
        with self.assertRaises(LLMResponseError):
            self.client.chat([{"role": "user", "content": "test"}])

    def test_timeout_is_distinct(self):
        MockHandler.mode = "delay"
        client = LocalLLMClient(
            base_url=self.base_url,
            model="mock-model",
            timeout_seconds=0.05,
        )
        with self.assertRaises(LLMTimeoutError):
            client.chat([{"role": "user", "content": "test"}])

    def test_connection_failure_is_distinct(self):
        client = LocalLLMClient(
            base_url="http://127.0.0.1:1",
            timeout_seconds=0.2,
        )
        with self.assertRaises(LLMConnectionError):
            client.chat([{"role": "user", "content": "test"}])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            self.client.chat([])
        with self.assertRaises(ValueError):
            self.client.chat(
                [{"role": "user", "content": "test"}],
                max_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
