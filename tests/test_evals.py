import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent.evals import (
    EvalCase,
    evaluate,
    format_comparison,
    format_summary,
    is_refusal,
    load_cases,
    main,
    percentile,
    run_evals,
)
from agent.llm_client import LLMConnectionError, LLMResponse
from agent.session_store import SessionStore


class FakeStream:
    def __init__(self, text, first=0.02, latency=0.06, token_logprobs=()):
        self._text = text
        self.text = ""
        self.token_logprobs = tuple(token_logprobs)
        self.model = "fake-model"
        self.prompt_tokens = 10
        self.completion_tokens = 4
        self.total_tokens = 14
        self.first_token_seconds = first
        self.latency_seconds = latency

    def __iter__(self):
        accumulated = ""
        for piece in self._text:
            accumulated += piece
            self.text = accumulated
            yield piece


class ScriptedClient:
    """Returns one scripted reply per request, in order, on either path."""

    def __init__(self, replies, token_logprobs=()):
        self.replies = list(replies)
        self.token_logprobs = tuple(token_logprobs)
        self.calls = []
        self.buffered_calls = 0

    def _take(self):
        return self.replies.pop(0) if self.replies else ""

    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        self.calls.append(list(messages))
        return FakeStream(self._take(), token_logprobs=self.token_logprobs)

    def chat(self, messages, *, temperature=0.0, max_tokens=256, logprobs=False):
        self.calls.append(list(messages))
        self.buffered_calls += 1
        return LLMResponse(
            text=self._take(),
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            latency_seconds=0.05,
            token_logprobs=self.token_logprobs if logprobs else (),
        )


class BrokenClient:
    def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
        raise LLMConnectionError("Could not connect to the inference server")

    def chat(self, messages, *, temperature=0.0, max_tokens=256):
        raise LLMConnectionError("Could not connect to the inference server")


class EvalCaseFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write_cases(self, payload):
        path = Path(self.temp_dir.name) / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_cases_with_checks(self):
        path = self.write_cases({
            "cases": [
                {
                    "id": "one",
                    "prompt": "say something",
                    "expect_contains": ["x"],
                    "expect_any": ["a", "b"],
                    "expect_absent": ["nope"],
                    "expect_refusal": True,
                }
            ]
        })

        cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "one")
        self.assertEqual(cases[0].expect_contains, ("x",))
        self.assertEqual(cases[0].expect_any, ("a", "b"))
        self.assertEqual(cases[0].expect_absent, ("nope",))
        self.assertIs(cases[0].expect_refusal, True)

    def test_accepts_a_bare_list(self):
        path = self.write_cases([{"id": "one", "prompt": "p"}])
        self.assertEqual(len(load_cases(path)), 1)

    def test_rejects_unknown_fields(self):
        path = self.write_cases([
            {"id": "one", "prompt": "p", "expect_contain": ["typo"]}
        ])
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_rejects_duplicate_ids(self):
        path = self.write_cases([
            {"id": "same", "prompt": "a"},
            {"id": "same", "prompt": "b"},
        ])
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_rejects_bad_expectation_types(self):
        path = self.write_cases([
            {"id": "one", "prompt": "p", "expect_contains": "not a list"}
        ])
        with self.assertRaises(ValueError):
            load_cases(path)

        path = self.write_cases([
            {"id": "one", "prompt": "p", "expect_refusal": "yes"}
        ])
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_rejects_an_invalid_regex(self):
        path = self.write_cases([
            {"id": "one", "prompt": "p", "forbid_regex": ["(unclosed"]}
        ])
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_rejects_empty_case_list(self):
        path = self.write_cases({"cases": []})
        with self.assertRaises(ValueError):
            load_cases(path)


class CheckTests(unittest.TestCase):
    def test_refusal_detection(self):
        self.assertTrue(is_refusal("I can't help with that."))
        self.assertTrue(is_refusal("I don't have access to your bank account."))
        self.assertTrue(
            is_refusal("That figure is not publicly available.")
        )
        self.assertTrue(is_refusal("I cannot verify a current number."))
        self.assertFalse(is_refusal("Use items[-1] to get the last element."))

    def test_contains_and_absent(self):
        case = EvalCase(
            id="c", prompt="p",
            expect_contains=("ValueError",),
            expect_absent=("Traceback",),
        )

        self.assertEqual(evaluate(case, "It raises ValueError."), ())
        missing = evaluate(case, "It raises TypeError.")
        self.assertTrue(any("missing 'ValueError'" in item for item in missing))

        leaked = evaluate(case, "ValueError: Traceback follows")
        self.assertTrue(any("contains 'Traceback'" in item for item in leaked))

    def test_expect_any_needs_one_alternative(self):
        case = EvalCase(id="c", prompt="p", expect_any=("x*x", "x**2"))

        self.assertEqual(evaluate(case, "use [x**2 for x in range(1, 6)]"), ())
        self.assertTrue(evaluate(case, "use pow(x, 2)"))

    def test_forbidden_patterns_fail_even_on_a_hedged_reply(self):
        case = EvalCase(
            id="c", prompt="p", expect_refusal=True,
            forbid_regex=(r"\$\s*\d",),
        )

        self.assertEqual(
            evaluate(case, "That figure is not publicly available."), ()
        )
        failures = evaluate(case, "His net worth is about $4.1 billion.")
        self.assertTrue(any("matches forbidden" in item for item in failures))
        self.assertTrue(any("expected a refusal" in item for item in failures))

    def test_refusal_expectations(self):
        must_refuse = EvalCase(id="c", prompt="p", expect_refusal=True)
        self.assertEqual(evaluate(must_refuse, "I cannot know that."), ())
        self.assertTrue(evaluate(must_refuse, "It is 12 dollars."))

        must_answer = EvalCase(id="c", prompt="p", expect_refusal=False)
        self.assertTrue(evaluate(must_answer, "I don't know."))

    def test_percentiles(self):
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([10.0], 0.95), 10.0)
        self.assertAlmostEqual(percentile([0.0, 100.0], 0.5), 50.0)
        self.assertAlmostEqual(percentile([0.0, 100.0], 0.95), 95.0)


class EvalRunTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = temp_dir
        self.store = SessionStore(Path(temp_dir.name) / "evals.sqlite3")

    def test_summary_counts_refusal_rate_and_latency(self):
        cases = [
            EvalCase(id="passes", prompt="q1", expect_contains=("keyword",)),
            EvalCase(id="fails", prompt="q2", expect_contains=("missing",)),
            EvalCase(id="refuses", prompt="q3", expect_refusal=True),
        ]
        client = ScriptedClient([
            "the keyword is here",
            "nothing useful",
            "I can't answer that.",
        ])

        summary = run_evals(cases, self.store, client, max_tokens=64)

        self.assertEqual(summary.passed, 2)
        self.assertEqual(summary.failed, 1)
        self.assertAlmostEqual(summary.refusal_rate, 1 / 3)
        self.assertGreater(summary.latency_p50, 0.0)
        self.assertGreaterEqual(summary.latency_p95, summary.latency_p50)

        failed = [r for r in summary.results if not r.passed][0]
        self.assertEqual(failed.case.id, "fails")
        self.assertTrue(failed.failures)

    def test_each_case_runs_in_a_fresh_session(self):
        cases = [
            EvalCase(id="first", prompt="first question"),
            EvalCase(id="second", prompt="second question"),
        ]
        client = ScriptedClient(["one", "two"])

        run_evals(cases, self.store, client, max_tokens=64)

        for call, case in zip(client.calls, cases):
            self.assertEqual(len(call), 2)
            self.assertEqual(call[0]["role"], "system")
            self.assertEqual(call[1]["content"], case.prompt)

    def test_request_failure_marks_the_case_failed(self):
        cases = [EvalCase(id="broken", prompt="q")]

        summary = run_evals(cases, self.store, BrokenClient(), max_tokens=64)

        self.assertEqual(summary.failed, 1)
        self.assertIn("case failed to run", summary.results[0].failures[0])
        self.assertEqual(summary.latency_p50, 0.0)

    def test_a_case_that_cannot_run_does_not_abort_the_run(self):
        cases = [
            EvalCase(id="too-big", prompt="word " * 400),
            EvalCase(id="fine", prompt="hello", expect_contains=("hi",)),
        ]

        summary = run_evals(
            cases, self.store, ScriptedClient(["hi"]),
            context_tokens=200, max_tokens=50,
        )

        self.assertEqual(len(summary.results), 2)
        self.assertEqual(summary.results[0].case.id, "too-big")
        self.assertFalse(summary.results[0].passed)
        self.assertIn("failed to run", summary.results[0].failures[0])
        self.assertTrue(summary.results[1].passed)

    def test_an_unexpected_crash_is_contained(self):
        class BoomThenOk(ScriptedClient):
            def chat_stream(self, messages, *, temperature=0.0, max_tokens=256):
                if not self.calls:
                    self.calls.append(list(messages))
                    raise RuntimeError("boom")
                return super().chat_stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                )

        cases = [EvalCase(id="boom", prompt="q"), EvalCase(id="ok", prompt="q")]

        summary = run_evals(cases, self.store, BoomThenOk(["fine"]))

        self.assertFalse(summary.results[0].passed)
        self.assertIn("case crashed: RuntimeError", summary.results[0].failures[0])
        self.assertTrue(summary.results[1].passed)
        self.assertEqual(summary.refusal_rate, 0.0)

    def test_gated_cases_are_counted_and_hedge_as_refusal(self):
        cases = [EvalCase(id="one", prompt="q", expect_contains=("guaranteed",))]

        summary = run_evals(
            cases,
            self.store,
            ScriptedClient(["answer"], token_logprobs=(-2.0,) * 5),
            confidence_threshold=0.55,
        )

        self.assertEqual(summary.gated, 1)
        self.assertEqual(summary.pass_rate, 0.0)
        self.assertEqual(summary.refusal_rate, 1.0)
        self.assertAlmostEqual(summary.results[0].confidence, 0.135, places=3)

    def test_comparison_reports_pass_and_refusal_deltas(self):
        case = EvalCase(id="greet", prompt="q", expect_contains=("hi there",))
        baseline = run_evals(cases=[case], store=self.store,
                             client=ScriptedClient(["hi there"]))
        gated = run_evals(
            cases=[case], store=self.store,
            client=ScriptedClient(["hi there"], token_logprobs=(-2.0,) * 5),
            confidence_threshold=0.55,
        )

        text = format_comparison(baseline, gated, 0.55)

        self.assertIn("confidence threshold 0.55", text)
        self.assertIn("pass rate", text)
        self.assertIn("100% -> 0%", text)
        self.assertIn("-100 pts", text)
        self.assertIn("refusal rate", text)
        self.assertIn("0% -> 100%", text)
        self.assertIn("+100 pts", text)
        self.assertIn("gated        0 -> 1", text)
        self.assertIn("broken       greet", text)

    def test_format_summary_reports_pass_fail_and_metrics(self):
        cases = [EvalCase(id="ok", prompt="q", expect_contains=("yes",))]
        summary = run_evals(cases, self.store, ScriptedClient(["yes"]),
                            max_tokens=64)

        text = format_summary(summary)

        self.assertIn("PASS  ok", text)
        self.assertIn("cases 1 | passed 1 | failed 0", text)
        self.assertIn("refusal rate 0%", text)
        self.assertIn("p50", text)
        self.assertIn("p95", text)


class EvalMainTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.dir = Path(temp_dir.name)

    def run_main(self, cases_payload, client, extra=None):
        cases_path = self.dir / "cases.json"
        if cases_payload is not None:
            cases_path.write_text(json.dumps(cases_payload), encoding="utf-8")
        argv = [
            "--cases", str(cases_path),
            "--db", str(self.dir / "evals.sqlite3"),
            "--max-tokens", "64",
        ] + (extra or [])

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv, client=client)
        return code, out.getvalue(), err.getvalue()

    def test_exit_zero_when_all_cases_pass(self):
        payload = [
            {"id": "greet", "prompt": "say hi", "expect_contains": ["hi"]}
        ]
        code, out, _ = self.run_main(payload, ScriptedClient(["hi there"]))

        self.assertEqual(code, 0)
        self.assertIn("cases 1 | passed 1 | failed 0", out)

    def test_exit_one_when_a_case_fails(self):
        payload = [
            {"id": "greet", "prompt": "say hi", "expect_contains": ["hello"]}
        ]
        code, out, _ = self.run_main(payload, ScriptedClient(["hi there"]))

        self.assertEqual(code, 1)
        self.assertIn("FAIL  greet", out)
        self.assertIn("missing 'hello'", out)

    def test_compare_gating_runs_twice_and_prints_deltas(self):
        payload = [
            {"id": "greet", "prompt": "say hi", "expect_contains": ["hi there"]}
        ]
        client = ScriptedClient(
            ["hi there", "hi there"], token_logprobs=(-2.0,) * 5
        )

        code, out, _ = self.run_main(
            payload, client, extra=["--compare-gating"]
        )

        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("=== baseline (gating off) ===", out)
        self.assertIn("=== gated (threshold 0.55) ===", out)
        self.assertIn("gated 1", out)
        self.assertIn("gating comparison", out)

    def test_invalid_confidence_threshold_exits_two(self):
        payload = [{"id": "greet", "prompt": "q"}]

        code, _, err = self.run_main(
            payload, ScriptedClient([]), extra=["--confidence-threshold", "2"]
        )

        self.assertEqual(code, 2)
        self.assertIn("--confidence-threshold", err)

    def test_only_filters_to_one_case(self):
        payload = [
            {"id": "first", "prompt": "q1", "expect_contains": ["one"]},
            {"id": "second", "prompt": "q2", "expect_contains": ["two"]},
        ]
        client = ScriptedClient(["two"])

        code, out, _ = self.run_main(
            payload, client, extra=["--only", "second"]
        )

        self.assertEqual(code, 0)
        self.assertIn("second", out)
        self.assertNotIn("first", out)
        self.assertEqual(len(client.calls), 1)

    def test_unknown_only_id_exits_two(self):
        payload = [{"id": "first", "prompt": "q1"}]

        code, _, err = self.run_main(
            payload, ScriptedClient([]), extra=["--only", "nope"]
        )

        self.assertEqual(code, 2)
        self.assertIn("no case with id", err)

    def test_missing_case_file_exits_two(self):
        code, _, err = self.run_main(None, ScriptedClient([]))

        self.assertEqual(code, 2)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
