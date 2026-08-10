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
from src.eval.metrics import compute_cer, compute_wer, term_recall
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

    lexicon_terms, lexicon_source = load_lexicon_terms()
    recall = term_recall(references, hypotheses, lexicon_terms)

    report = {
        "variant": args.variant,
        "model": settings.stt_baseline_model
        if args.variant == "openai"
        else settings.whisper_model,
        "adapter_path": args.adapter_path if args.variant.startswith("finetuned") else None,
        "n_clips": len(records),
        "wer": round(compute_wer(references, hypotheses), 4),
        "cer": round(compute_cer(references, hypotheses), 4),
        "term_recall": round(recall["recall"], 4) if recall["recall"] is not None else None,
        "term_occurrences": recall["n_ref_occurrences"],
        "term_hits": recall["n_hits"],
        "lexicon_source": lexicon_source,
        "per_term": recall["per_term"],
        "created_at": datetime.now(UTC).isoformat(),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"report_{args.variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"{args.variant}: WER {report['wer']:.2%}  CER {report['cer']:.2%}  "
        f"term recall {report['term_recall'] if report['term_recall'] is not None else 'n/a'}"
    )
    print(f"Wrote {out_path}")
    return out_path


def compare() -> None:
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

    header = "| Variant | Model | Clips | WER | CER | Term recall |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for r in rows:
        tr = f"{r['term_recall']:.2%}" if r["term_recall"] is not None else "n/a"
        model = r["model"].split("/")[-1] + (" + LoRA" if r["adapter_path"] else "")
        if r["variant"] == "finetuned_biased":
            model += " + hotwords"
        lines.append(
            f"| {r['variant']} | {model} | {r['n_clips']} "
            f"| {r['wer']:.2%} | {r['cer']:.2%} | {tr} |"
        )

    table = "\n".join(lines)
    out_path = EVAL_DIR / "comparison.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nWrote {out_path}")


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
