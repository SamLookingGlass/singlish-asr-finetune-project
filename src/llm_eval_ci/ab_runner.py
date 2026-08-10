"""Compare two prompt versions (A/B test) against the hand-labeled gold set.

Both prompt versions classify every term in the gold set, and each is
scored against the hand labels: precision, recall, and F1 (where "hotword"
is the positive class). Scores, latency, and estimated cost are logged to
MLflow.

A McNemar test then checks whether the difference between the two versions
is statistically significant, or small enough to be luck.

Outputs:
- data/eval/ab_report.json        (read by gate.py in CI)
- data/eval/mlflow_summary.json   (summary kept in the repo; the MLflow db is gitignored)

Run:
    python -m src.llm_eval_ci.ab_runner              # full gold set, v1 vs v2
    python -m src.llm_eval_ci.ab_runner --limit 40   # cheaper smoke run
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime

from src.config import EVAL_DIR, LEXICON_DIR, settings
from src.lexicon.llm_curate import classify_term, get_client
from src.lexicon.prompts import PROMPT_VERSIONS

AB_REPORT = EVAL_DIR / "ab_report.json"
MLFLOW_SUMMARY = EVAL_DIR / "mlflow_summary.json"

# Approximate gpt-4o-mini pricing (USD per 1M tokens) for cost estimates only.
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60


def evaluate_predictions(gold_labels: list[str], predictions: list[str]) -> dict:
    """Score predictions against the gold labels ("hotword" is the positive class).

    Returns precision, recall, F1, accuracy, and a per-item list of which
    answers were correct (the McNemar test needs that list).
    """
    tp = fp = fn = tn = 0
    correct: list[bool] = []
    for gold, pred in zip(gold_labels, predictions):
        is_correct = gold == pred
        correct.append(is_correct)
        if pred == "hotword" and gold == "hotword":
            tp += 1
        elif pred == "hotword" and gold == "noise":
            fp += 1
        elif pred == "noise" and gold == "hotword":
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(sum(correct) / len(correct), 4) if correct else 0.0,
        "n": len(correct),
        "correct": correct,
    }


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact McNemar test p-value.

    b = items only the baseline got right, c = items only the candidate got
    right. If the two prompts were equally good, each disagreement would go
    either way with 50/50 odds; the p-value says how surprising the
    observed split is under that assumption. Small p (< 0.05) means the
    difference is probably real.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def run_version(version: str, gold: list[dict], model: str | None) -> dict:
    """Classify every gold term with one prompt version and score the results."""
    client = get_client()
    predictions: list[str] = []
    latencies: list[float] = []
    prompt_tokens = completion_tokens = 0
    for i, item in enumerate(gold):
        decision = classify_term(
            client, item["term"], examples=None, prompt_version=version, model=model
        )
        predictions.append(decision["label"])
        latencies.append(decision["latency_s"])
        prompt_tokens += decision["prompt_tokens"]
        completion_tokens += decision["completion_tokens"]
        if (i + 1) % 25 == 0:
            print(f"  [{version}] {i + 1}/{len(gold)}")

    scores = evaluate_predictions([g["label"] for g in gold], predictions)
    scores["avg_latency_s"] = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    scores["est_cost_usd"] = round(
        prompt_tokens / 1e6 * PRICE_INPUT_PER_M + completion_tokens / 1e6 * PRICE_OUTPUT_PER_M, 4
    )
    scores["predictions"] = predictions
    return scores


def main() -> None:
    from src.llm_eval_ci.mlflow_logger import log_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs=2, default=["v1", "v2"],
                        metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--gold", default=str(LEXICON_DIR / "gold_labels.json"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N terms")
    args = parser.parse_args()

    for v in args.versions:
        if v not in PROMPT_VERSIONS:
            raise SystemExit(f"unknown prompt version '{v}' (available: {sorted(PROMPT_VERSIONS)})")

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)["labels"]
    if args.limit is not None:
        gold = gold[: args.limit]
    print(f"Evaluating prompt versions {args.versions} on {len(gold)} gold-labeled terms")

    model = args.model or settings.llm_model
    results: dict[str, dict] = {}
    run_ids: dict[str, str] = {}
    for version in args.versions:
        print(f"Running prompt {version}...")
        scores = run_version(version, gold, model)
        results[version] = scores
        run_ids[version] = log_run(
            prompt_version=version,
            params={"model": model, "n_gold": len(gold), "positive_class": "hotword"},
            metrics={
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "accuracy": scores["accuracy"],
                "avg_latency_s": scores["avg_latency_s"],
                "est_cost_usd": scores["est_cost_usd"],
            },
        )
        print(
            f"  {version}: P {scores['precision']:.3f}  R {scores['recall']:.3f}  "
            f"F1 {scores['f1']:.3f}  acc {scores['accuracy']:.3f}"
        )

    baseline, candidate = args.versions
    b = sum(
        1
        for i in range(len(gold))
        if results[baseline]["correct"][i] and not results[candidate]["correct"][i]
    )
    c = sum(
        1
        for i in range(len(gold))
        if not results[baseline]["correct"][i] and results[candidate]["correct"][i]
    )
    p_value = mcnemar_pvalue(b, c)
    significance = {
        "test": "mcnemar_exact",
        "baseline_only_correct": b,
        "candidate_only_correct": c,
        "p_value": round(p_value, 4),
        "significant_at_0.05": p_value < 0.05,
    }
    print(f"McNemar: b={b} c={c} p={p_value:.4f} significant={p_value < 0.05}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "model": model,
        "n_gold": len(gold),
        "results": {
            v: {k: val for k, val in scores.items() if k not in ("correct", "predictions")}
            for v, scores in results.items()
        },
        "significance": significance,
        "mlflow_run_ids": run_ids,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with open(AB_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(MLFLOW_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "prompt-regression",
                "runs": [
                    {"prompt_version": v, "mlflow_run_id": run_ids[v], **report["results"][v]}
                    for v in args.versions
                ],
                "created_at": report["created_at"],
            },
            f,
            indent=2,
        )
    print(f"Wrote {AB_REPORT} and {MLFLOW_SUMMARY}")


if __name__ == "__main__":
    main()
