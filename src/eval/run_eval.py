"""Benchmark the ASR systems on the test split and build a comparison table.

The four systems ("variants") being compared:
- base              whisper-large-v3-turbo as released, no adaptation
- finetuned         base model + our LoRA adapter
- finetuned_biased  fine-tuned model + hotword prompt during decoding
- openai            commercial baseline (gpt-4o-transcribe API)

Each run scores one variant (WER, CER, term recall) and writes
data/eval/report_<variant>.json. The ``--compare`` flag gathers every
report into one markdown table (data/eval/comparison.md) for the README.

Examples:
    python -m src.eval.run_eval --variant base
    python -m src.eval.run_eval --variant openai --limit 500
    python -m src.eval.run_eval --compare
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import EVAL_DIR, LEXICON_DIR, MANIFEST_DIR, settings
from src.eval.figures import write_wer_figure
from src.eval.metrics import (
    bootstrap_cer_ci,
    bootstrap_wer_ci,
    compute_cer,
    compute_wer,
    stratify_by_duration,
    term_recall,
)
from src.lexicon.mine_hotwords import SEED_TERMS

VARIANTS = ("base", "finetuned", "finetuned_biased", "openai")


def load_lexicon_terms() -> tuple[list[str], str]:
    """Load the best lexicon we have: curated first, then mined, then the seed list."""
    curated = LEXICON_DIR / "curated_lexicon.json"
    mined = LEXICON_DIR / "mined_candidates.json"
    if curated.exists():
        with open(curated, encoding="utf-8") as f:
            return [t["term"] for t in json.load(f)["terms"]], "curated_lexicon.json"
    if mined.exists():
        with open(mined, encoding="utf-8") as f:
            return [c["term"] for c in json.load(f)], "mined_candidates.json"
    return sorted(SEED_TERMS), "seed_terms (run mining + curation for the full lexicon)"


def get_hypotheses(records: list[dict], variant: str, args: argparse.Namespace) -> list[str]:
    """Transcribe the test clips with whichever system the variant names."""
    if variant == "openai":
        from src.eval.baselines import transcribe_openai

        results = transcribe_openai(records, limit=None)
        by_id = {r["id"]: r["hypothesis"] for r in results}
        return [by_id[r["id"]] for r in records]

    from src.eval.decode import LocalWhisperTranscriber

    hotwords = None
    if variant == "finetuned_biased":
        hotwords, source = load_lexicon_terms()
        print(f"contextual biasing with {len(hotwords)} terms from {source}")

    adapter = args.adapter_path if variant.startswith("finetuned") else None
    if variant.startswith("finetuned") and not adapter:
        raise SystemExit(f"--adapter-path is required for variant '{variant}'")

    transcriber = LocalWhisperTranscriber(
        adapter_path=adapter, hotwords=hotwords, batch_size=args.batch_size
    )
    return transcriber.transcribe([r["audio_path"] for r in records])


def run_variant(args: argparse.Namespace) -> Path:
    """Score one variant on the test split and write its report JSON."""
    with open(args.test_manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if args.limit is not None:
        records = records[: args.limit]

    references = [r["transcript"] for r in records]
    hypotheses = get_hypotheses(records, args.variant, args)

    wer = round(compute_wer(references, hypotheses), 4)
    cer = round(compute_cer(references, hypotheses), 4)
    wer_ci = bootstrap_wer_ci(references, hypotheses)
    cer_ci = bootstrap_cer_ci(references, hypotheses)
    stratified = stratify_by_duration(records, references, hypotheses)

    # Always measure the standard Singlish seed terms
    seed_recall = term_recall(references, hypotheses, sorted(SEED_TERMS))

    lexicon_terms, lexicon_source = load_lexicon_terms()
    has_full_lexicon = lexicon_source != "seed_terms (run mining + curation for the full lexicon)"
    if has_full_lexicon:
        full_recall = term_recall(references, hypotheses, lexicon_terms)
    else:
        full_recall = None

    report = {
        "variant": args.variant,
        "model": settings.stt_baseline_model
        if args.variant == "openai"
        else settings.whisper_model,
        "adapter_path": args.adapter_path if args.variant.startswith("finetuned") else None,
        "n_clips": len(records),
        "wer": wer,
        "wer_ci_95": wer_ci,
        "cer": cer,
        "cer_ci_95": cer_ci,
        "term_recall": round(seed_recall["recall"], 4) if seed_recall["recall"] is not None else None,
        "term_occurrences": seed_recall["n_ref_occurrences"],
        "term_hits": seed_recall["n_hits"],
        "lexicon_source": "seed_terms",
        "per_term": seed_recall["per_term"],
        "stratification_by_duration": stratified,
        "created_at": datetime.now(UTC).isoformat(),
    }

    if full_recall is not None:
        report["full_lexicon_term_recall"] = (
            round(full_recall["recall"], 4) if full_recall["recall"] is not None else None
        )
        report["full_lexicon_source"] = lexicon_source
        report["full_lexicon_occurrences"] = full_recall["n_ref_occurrences"]
        report["full_lexicon_hits"] = full_recall["n_hits"]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"report_{args.variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    ci_info = f" (95% CI: [{wer_ci[0]:.2%}, {wer_ci[1]:.2%}])"
    print(
        f"{args.variant}: WER {report['wer']:.2%}{ci_info}  CER {report['cer']:.2%}  "
        f"term recall {report['term_recall'] if report['term_recall'] is not None else 'n/a'}"
    )
    print(f"Wrote {out_path}")
    return out_path


def compare() -> str:
    """Collect all report files into one markdown comparison table."""
    rows = []
    for variant in VARIANTS:
        path = EVAL_DIR / f"report_{variant}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit("no reports found - run --variant evals first")

    has_ci = any("wer_ci_95" in r for r in rows)
    if has_ci:
        header = "| Variant | Model | Clips | WER (95% CI) | CER (95% CI) | Term recall |"
    else:
        header = "| Variant | Model | Clips | WER | CER | Term recall |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        tr = f"{r['term_recall']:.2%}" if r["term_recall"] is not None else "n/a"
        model = r["model"].split("/")[-1] + (" + LoRA" if r.get("adapter_path") else "")
        if r["variant"] == "finetuned_biased":
            model += " + hotwords"
        if has_ci and "wer_ci_95" in r and "cer_ci_95" in r:
            wer_display = f"{r['wer']:.2%} [{r['wer_ci_95'][0]:.1%}, {r['wer_ci_95'][1]:.1%}]"
            cer_display = f"{r['cer']:.2%} [{r['cer_ci_95'][0]:.1%}, {r['cer_ci_95'][1]:.1%}]"
        else:
            wer_display = f"{r['wer']:.2%}"
            cer_display = f"{r['cer']:.2%}"
        lines.append(
            f"| {r['variant']} | {model} | {r['n_clips']} "
            f"| {wer_display} | {cer_display} | {tr} |"
        )

    table = "\n".join(lines)
    out_path = EVAL_DIR / "comparison.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    figure = write_wer_figure(rows)
    print(table)
    print(f"\nWrote {out_path}")
    print(f"Wrote {figure}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--test-manifest", default=str(MANIFEST_DIR / "test.jsonl"))
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap clips (recommended for openai)")
    parser.add_argument("--compare", action="store_true", help="collate existing reports")
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.variant:
        run_variant(args)
    else:
        parser.error("provide --variant or --compare")


if __name__ == "__main__":
    main()
