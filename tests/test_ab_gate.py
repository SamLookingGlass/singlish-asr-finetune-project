import json

import pytest

from src.llm_eval_ci.ab_runner import evaluate_predictions, mcnemar_pvalue
from src.llm_eval_ci.gate import _from_report, check_regression


def test_evaluate_predictions_perfect():
    gold = ["hotword", "noise", "hotword"]
    scores = evaluate_predictions(gold, list(gold))
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["accuracy"] == 1.0


def test_evaluate_predictions_mixed():
    gold = ["hotword", "hotword", "noise", "noise"]
    pred = ["hotword", "noise", "hotword", "noise"]
    scores = evaluate_predictions(gold, pred)
    # tp=1 fp=1 fn=1 tn=1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    assert scores["accuracy"] == 0.5
    assert scores["correct"] == [True, False, False, True]


def test_mcnemar_no_disagreements():
    assert mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_symmetric_disagreements_not_significant():
    assert mcnemar_pvalue(5, 5) > 0.05


def test_mcnemar_lopsided_disagreements_significant():
    assert mcnemar_pvalue(0, 15) < 0.05


def test_mcnemar_capped_at_one():
    assert mcnemar_pvalue(3, 3) <= 1.0


def test_check_regression_pass_when_candidate_better():
    verdict = check_regression(baseline_f1=0.80, candidate_f1=0.90, threshold=0.05)
    assert verdict["ok"] is True
    assert verdict["drop"] == -0.10


def test_check_regression_pass_within_threshold():
    verdict = check_regression(baseline_f1=0.80, candidate_f1=0.77, threshold=0.05)
    assert verdict["ok"] is True


def test_check_regression_fail_beyond_threshold():
    verdict = check_regression(baseline_f1=0.80, candidate_f1=0.70, threshold=0.05)
    assert verdict["ok"] is False
    assert verdict["drop"] == pytest.approx(0.10)


def test_from_report_reads_f1_values(tmp_path):
    report = {
        "baseline": "v1",
        "candidate": "v2",
        "results": {"v1": {"f1": 0.8}, "v2": {"f1": 0.85}},
    }
    path = tmp_path / "ab_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    baseline, baseline_f1, candidate, candidate_f1 = _from_report(str(path))
    assert (baseline, candidate) == ("v1", "v2")
    assert (baseline_f1, candidate_f1) == (0.8, 0.85)


def test_gate_end_to_end_via_report(tmp_path, monkeypatch, capsys):
    """gate.main() exits 1 on a regression and 0 otherwise."""
    import sys

    from src.llm_eval_ci import gate

    report = {
        "baseline": "v1",
        "candidate": "v2",
        "results": {"v1": {"f1": 0.90}, "v2": {"f1": 0.70}},
    }
    path = tmp_path / "ab_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["gate", "--report", str(path), "--threshold", "0.05"]
    )
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 1
    assert "GATE FAILED" in capsys.readouterr().out

    # relaxed threshold -> passes (main returns without SystemExit)
    monkeypatch.setattr(
        sys, "argv", ["gate", "--report", str(path), "--threshold", "0.5"]
    )
    gate.main()
    assert "GATE PASSED" in capsys.readouterr().out
