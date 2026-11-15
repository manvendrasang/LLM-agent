from __future__ import annotations

import argparse
import json
import re
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
    "not confident",
    "not sure enough",
)

CASE_FIELDS = {
    "id",
    "prompt",
    "expect_contains",
    "expect_any",
    "expect_absent",
    "expect_refusal",
    "forbid_regex",
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expect_contains: tuple[str, ...] = ()
    expect_any: tuple[str, ...] = ()
    expect_absent: tuple[str, ...] = ()
    expect_refusal: bool | None = None
    # Patterns the reply must not match, e.g. a dollar figure the model
    # cannot actually know.
    forbid_regex: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    reply: str
    passed: bool
    failures: tuple[str, ...]
    refused: bool
    total_ms: float
    first_token_ms: float
    gated: bool = False
    confidence: float | None = None


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
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed / len(self.results)

    @property
    def gated(self) -> int:
        return sum(1 for result in self.results if result.gated)

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


def _regex_tuple(raw: dict, key: str, case_id: str) -> tuple[str, ...]:
    patterns = _string_tuple(raw, key, case_id)
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"case {case_id!r}: {key} has an invalid regex {pattern!r}: {exc}"
            ) from exc
    return patterns


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
        forbid_regex=_regex_tuple(raw, "forbid_regex", case_id),
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

    for pattern in case.forbid_regex:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            failures.append(f"matches forbidden {pattern!r}: {match.group(0)!r}")

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
    confidence_threshold: float | None = None,
) -> CaseResult:
    try:
        chat = ChatSession(
            store,
            client,
            system_prompt=system_prompt,
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )
        result = chat.send(case.prompt)
    except (LLMClientError, ValueError) as exc:
        # A case that cannot run is a failed case, never a dead run: the rest
        # of the suite still tells us something.
        return CaseResult(
            case=case,
            reply="",
            passed=False,
            failures=(f"case failed to run: {type(exc).__name__}: {exc}",),
            refused=False,
            total_ms=0.0,
            first_token_ms=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the run
        return CaseResult(
            case=case,
            reply="",
            passed=False,
            failures=(f"case crashed: {type(exc).__name__}: {exc}",),
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
        # A gated answer is a non-answer by construction, whatever its wording.
        refused=is_refusal(result.text) or result.timings.gated,
        total_ms=result.timings.total_ms,
        first_token_ms=result.timings.first_token_ms,
        gated=result.timings.gated,
        confidence=result.timings.confidence,
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
        confidence = (
            f"conf {result.confidence:.2f}  "
            if result.confidence is not None
            else ""
        )
        lines.append(
            f"{mark}  {result.case.id:<24} {result.total_ms:7.0f} ms  "
            f"{confidence}{preview}"
        )
        for failure in result.failures:
            lines.append(f"      - {failure}")

    lines.append("")
    lines.append(
        f"cases {len(summary.results)} | passed {summary.passed} | "
        f"failed {summary.failed}"
    )
    gated = f" | gated {summary.gated}" if summary.gated else ""
    lines.append(
        f"refusal rate {summary.refusal_rate:.0%} | "
        f"p50 {summary.latency_p50:.0f} ms | p95 {summary.latency_p95:.0f} ms"
        f"{gated}"
    )
    return "\n".join(lines)


def _points(before: float, after: float) -> str:
    return f"{round((after - before) * 100):+d} pts"


def format_comparison(
    baseline: EvalSummary,
    gated: EvalSummary,
    threshold: float,
) -> str:
    """Side-by-side deltas between an ungated and a gated run."""
    by_id = {result.case.id: result.passed for result in baseline.results}
    fixed = [
        result.case.id
        for result in gated.results
        if result.passed and not by_id.get(result.case.id, False)
    ]
    broken = [
        result.case.id
        for result in gated.results
        if not result.passed and by_id.get(result.case.id, False)
    ]

    lines = [f"gating comparison (confidence threshold {threshold:g})"]
    lines.append(
        f"  pass rate    {baseline.pass_rate:.0%} -> {gated.pass_rate:.0%}  "
        f"({_points(baseline.pass_rate, gated.pass_rate)})"
    )
    lines.append(
        f"  refusal rate {baseline.refusal_rate:.0%} -> "
        f"{gated.refusal_rate:.0%}  "
        f"({_points(baseline.refusal_rate, gated.refusal_rate)})"
    )
    lines.append(f"  gated        {baseline.gated} -> {gated.gated}")
    lines.append(
        f"  p50          {baseline.latency_p50:.0f} ms -> "
        f"{gated.latency_p50:.0f} ms  "
        f"({gated.latency_p50 - baseline.latency_p50:+.0f} ms)"
    )
    if fixed:
        lines.append("  fixed        " + ", ".join(fixed))
    if broken:
        lines.append("  broken       " + ", ".join(broken))
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
    parser.add_argument(
        "--confidence-threshold", type=float, default=None, metavar="P",
        help=(
            "hedge instead of answering when the opening tokens' "
            "geometric-mean probability is below P"
        ),
    )
    parser.add_argument(
        "--compare-gating", action="store_true",
        help=(
            "run the suite ungated and gated, then print the pass-rate and "
            "refusal-rate deltas (uses --confidence-threshold, default 0.55)"
        ),
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

    threshold = args.confidence_threshold
    if threshold is not None and not (0.0 < threshold <= 1.0):
        print("error: --confidence-threshold must be in (0, 1]", file=sys.stderr)
        return 2

    store = SessionStore(args.db)
    settings = dict(
        system_prompt=args.system,
        context_tokens=args.context_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.compare_gating:
        threshold = 0.55 if threshold is None else threshold
        baseline = run_evals(cases, store, client, **settings)
        gated = run_evals(
            cases, store, client, confidence_threshold=threshold, **settings
        )
        print("=== baseline (gating off) ===")
        print(format_summary(baseline))
        print()
        print(f"=== gated (threshold {threshold:g}) ===")
        print(format_summary(gated))
        print()
        print(format_comparison(baseline, gated, threshold))
        return 0 if gated.failed == 0 else 1

    summary = run_evals(
        cases, store, client, confidence_threshold=threshold, **settings
    )
    print(format_summary(summary))
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
