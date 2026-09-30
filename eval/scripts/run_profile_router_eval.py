"""Evaluate the configured Profile router against fixed, synthetic user turns."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from harness_code_agent.profiles.router import (
    LLM_ROUTE_MIN_CONFIDENCE,
    JevRouteClassifier,
    LlmRouteResult,
    route_profile_for_turn,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "tests/fixtures/profile_routing_cases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/local_results/profile_router_jev.json")
    parser.add_argument("--interval", type=float, default=5.2, help="Minimum seconds between requests (12/minute gateway limit)")
    parser.add_argument("--replay", type=Path, help="Apply the current routing policy to saved real model decisions without new API calls")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be nonnegative")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    saved_rows = {}
    if args.replay:
        saved_rows = {row["case"]["id"]: row for row in json.loads(args.replay.read_text(encoding="utf-8"))["results"]}
        if set(saved_rows) != {case["id"] for case in cases}:
            parser.error("Replay must contain exactly the configured cases")
    rows = []
    classifier = JevRouteClassifier()
    next_request_at = 0.0
    try:
        for case in cases:
            classifier_for_case = classifier
            if args.replay:
                saved = saved_rows[case["id"]]
                if saved["case"] != case:
                    parser.error(f'Replay input differs for {case["id"]}')
                original = saved["decision"]
                captured = LlmRouteResult(
                    profile_name=original["matched_profile"], confidence=original["llm_confidence"],
                    reason=original["reason"], provider=original["llm_provider"], model=original["llm_model"],
                    failure_type="" if original["failure_type"] == "low_confidence" else original["failure_type"],
                    probabilities=original["probabilities"],
                )
                classifier_for_case = lambda captured=captured, **_: captured
            else:
                time.sleep(max(0.0, next_request_at - time.monotonic()))
                next_request_at = time.monotonic() + args.interval
            decision = route_profile_for_turn(
                case["prompt"], current_profile=case["current_profile"],
                previous_user_task=case.get("previous_user_task", ""),
                previous_assistant_text=case.get("previous_assistant_text", ""),
                llm_classifier=classifier_for_case,
            )
            passed = (
                decision.profile_name == case["expected_profile"]
                and decision.matched_profile == case["expected_matched_profile"]
                and decision.action == case["expected_action"]
                and not decision.fallback_used
            )
            rows.append({"case": case, "decision": asdict(decision), "passed": passed})
            print(f'{case["id"]}: {"PASS" if passed else "FAIL"} {decision.matched_profile} '
                  f'confidence={decision.confidence:.3f} elapsed={decision.elapsed_ms:.0f}ms failure={decision.failure_type or "none"}', flush=True)
            if decision.failure_type == "rate_limit":
                break
    finally:
        classifier.close()
    durations = [row["decision"]["elapsed_ms"] for row in rows]
    summary = {
        "confidence_threshold": LLM_ROUTE_MIN_CONFIDENCE,
        "source": "replay" if args.replay else "live",
        "replay_path": str(args.replay) if args.replay else None,
        "total": len(cases), "evaluated": len(rows), "passed": sum(row["passed"] for row in rows),
        "mean_ms": round(statistics.mean(durations), 1) if not args.replay else None,
        "median_ms": round(statistics.median(durations), 1) if not args.replay else None,
        "max_ms": round(max(durations), 1) if not args.replay else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
