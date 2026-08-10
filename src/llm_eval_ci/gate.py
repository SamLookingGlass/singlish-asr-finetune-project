"""Quality gate for prompt changes: fail the build if the new prompt is worse.

Reads the A/B report that ab_runner.py just wrote, compares the candidate
prompt's F1 score against the baseline prompt's F1 score, and exits with
code 1 if the drop is bigger than the threshold. In GitHub Actions, a
non-zero exit blocks the merge.

Run:
    python -m src.llm_eval_ci.gate --report data/eval/ab_report.json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import EVAL_DIR, settings


def check_regression(baseline_f1: float, candidate_f1: float, threshold: float) -> dict:
    """Return {"ok": bool, "drop": float}. A positive drop means the candidate is worse."""
    drop = baseline_f1 - candidate_f1
    return {"ok": drop <= threshold, "drop": round(drop, 4)}


def _from_report(report_path: str) -> tuple[str, float, str, float]:
    """Read baseline/candidate names and their F1 scores from the A/B report."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    baseline, candidate = report["baseline"], report["candidate"]
    return (
        baseline,
        report["results"][baseline]["f1"],
        candidate,
        report["results"][candidate]["f1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(EVAL_DIR / "ab_report.json"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.regression_threshold_f1,
        help="largest F1 drop we accept, in absolute points (default from config)",
    )
    args = parser.parse_args()

    baseline, baseline_f1, candidate, candidate_f1 = _from_report(args.report)

    verdict = check_regression(baseline_f1, candidate_f1, args.threshold)
    print(
        f"baseline {baseline}: F1 {baseline_f1:.4f} | candidate {candidate}: "
        f"F1 {candidate_f1:.4f} | drop {verdict['drop']:+.4f} "
        f"(threshold {args.threshold})"
    )
    if not verdict["ok"]:
        print(f"GATE FAILED: candidate '{candidate}' regresses F1 beyond the threshold.")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
