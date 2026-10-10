from src.eval.figures import (
    write_bar_svg,
    write_corpus_figure,
    write_loss_curve_figure,
    write_wer_figure,
)


def test_corpus_and_wer_svgs(tmp_path, monkeypatch):
    from src.eval import figures

    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)

    corpus = write_corpus_figure(
        {
            "train": {"hours": 11.97},
            "val": {"hours": 1.66},
            "test": {"hours": 1.54},
        }
    )
    wer = write_wer_figure(
        [
            {"variant": "base", "wer": 0.21},
            {"variant": "finetuned", "wer": 0.16},
        ]
    )
    text = corpus.read_text(encoding="utf-8")
    assert corpus.name == "corpus_hours.svg"
    assert "11.97 h" in text
    assert "finetuned" in wer.read_text(encoding="utf-8")
    assert "16.00%" in wer.read_text(encoding="utf-8")

    custom = write_bar_svg(tmp_path / "custom.svg", "Hours", ["a"], [1.5], " h")
    assert "1.50 h" in custom.read_text(encoding="utf-8")


def test_compare_generates_markdown_table_and_figure(tmp_path, monkeypatch):
    import json

    from src.eval import figures, run_eval

    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)

    base_report = {
        "variant": "base",
        "model": "openai/whisper-large-v3-turbo",
        "adapter_path": None,
        "n_clips": 10,
        "wer": 0.25,
        "wer_ci_95": [0.22, 0.28],
        "cer": 0.15,
        "cer_ci_95": [0.13, 0.17],
        "term_recall": 0.50,
    }
    ft_report = {
        "variant": "finetuned",
        "model": "openai/whisper-large-v3-turbo",
        "adapter_path": "checkpoints/mock",
        "n_clips": 10,
        "wer": 0.18,
        "wer_ci_95": [0.15, 0.21],
        "cer": 0.10,
        "cer_ci_95": [0.08, 0.12],
        "term_recall": 0.60,
    }
    (tmp_path / "report_base.json").write_text(json.dumps(base_report), encoding="utf-8")
    (tmp_path / "report_finetuned.json").write_text(json.dumps(ft_report), encoding="utf-8")

    table = run_eval.compare()
    assert "| Variant | Model | Clips | WER (95% CI) | CER (95% CI) | Term recall |" in table
    assert "25.00% [22.0%, 28.0%]" in table
    assert "18.00% [15.0%, 21.0%]" in table
    assert (tmp_path / "comparison.md").is_file()


def test_loss_curve_figure(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from src.eval import figures

    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)

    fig_path = write_loss_curve_figure()
    assert fig_path.name == "loss_curve.svg"
    assert fig_path.is_file()

    content = fig_path.read_text(encoding="utf-8")
    assert "Optimal Stopping (Step 555)" in content
    assert "Validation Loss" in content
    assert "Training Loss" in content
    assert "Eval Loss: 0.53" in content

    # Check XML well-formedness
    tree = ET.fromstring(content)
    assert tree.tag.endswith("svg")
