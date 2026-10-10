"""Evaluation metrics: WER, CER, and domain-term recall.

All text is normalized first (lowercase, no punctuation, single spaces),
so two systems that only differ in formatting get the same score.

- WER (word error rate): fraction of words the system got wrong.
- CER (character error rate): same idea, counted per character.
- Term recall: of all the lexicon terms the speaker actually said,
  how many did the system transcribe? This isolates how well a system
  handles the Singlish domain terms specifically.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation (keep apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer

    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return jiwer.wer(refs, hyps)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer

    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    return jiwer.cer(refs, hyps)


def _count_occurrences(term: str, text: str) -> int:
    """Count whole-word matches of a term (which may be multi-word) in text."""
    pattern = r"\b" + re.escape(term) + r"\b"
    return len(re.findall(pattern, text))


def term_recall(
    references: list[str],
    hypotheses: list[str],
    terms: list[str],
) -> dict:
    """How many spoken lexicon terms made it into the transcriptions.

    For each clip: count how often each term appears in the reference, then
    check how many of those appearances also show up in the hypothesis.
    Hits are capped at the reference count, so a system that repeats a term
    it never heard does not gain extra credit.

    Returns {"recall", "n_ref_occurrences", "n_hits", "per_term": {...}}.
    Recall is None if no lexicon term appears in any reference.
    """
    norm_terms = sorted({normalize_text(t) for t in terms if normalize_text(t)})
    per_term: dict[str, dict[str, int]] = {t: {"ref": 0, "hit": 0} for t in norm_terms}

    total_ref = total_hit = 0
    for ref, hyp in zip(references, hypotheses):
        ref_n = normalize_text(ref)
        hyp_n = normalize_text(hyp)
        for term in norm_terms:
            n_ref = _count_occurrences(term, ref_n)
            if n_ref == 0:
                continue
            n_hit = min(n_ref, _count_occurrences(term, hyp_n))
            per_term[term]["ref"] += n_ref
            per_term[term]["hit"] += n_hit
            total_ref += n_ref
            total_hit += n_hit

    return {
        "recall": (total_hit / total_ref) if total_ref else None,
        "n_ref_occurrences": total_ref,
        "n_hits": total_hit,
        "per_term": {t: c for t, c in per_term.items() if c["ref"] > 0},
    }


def bootstrap_wer_ci(
    references: list[str],
    hypotheses: list[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate non-parametric bootstrap (lower, upper) confidence interval for WER."""
    import jiwer
    import numpy as np

    if not references:
        return 0.0, 0.0
    if len(references) == 1:
        score = compute_wer(references, hypotheses)
        return score, score

    edits: list[int] = []
    ref_lens: list[int] = []
    for r, h in zip(references, hypotheses):
        r_norm = normalize_text(r)
        h_norm = normalize_text(h)
        n_ref = len(r_norm.split())
        ref_lens.append(n_ref)
        if n_ref == 0:
            edits.append(len(h_norm.split()))
        else:
            out = jiwer.process_words(r_norm, h_norm)
            edits.append(out.substitutions + out.deletions + out.insertions)

    edits_arr = np.array(edits, dtype=np.int64)
    lens_arr = np.array(ref_lens, dtype=np.int64)

    rng = np.random.default_rng(seed)
    n = len(references)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    sample_edits = edits_arr[idx].sum(axis=1)
    sample_lens = lens_arr[idx].sum(axis=1)
    valid = sample_lens > 0
    if not np.any(valid):
        return 0.0, 0.0

    sample_wers = sample_edits[valid] / sample_lens[valid]
    lower = float(np.percentile(sample_wers, 100 * (alpha / 2)))
    upper = float(np.percentile(sample_wers, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def bootstrap_cer_ci(
    references: list[str],
    hypotheses: list[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate non-parametric bootstrap (lower, upper) confidence interval for CER."""
    import jiwer
    import numpy as np

    if not references:
        return 0.0, 0.0
    if len(references) == 1:
        score = compute_cer(references, hypotheses)
        return score, score

    edits: list[int] = []
    ref_lens: list[int] = []
    for r, h in zip(references, hypotheses):
        r_norm = normalize_text(r)
        h_norm = normalize_text(h)
        n_ref = len(r_norm)
        ref_lens.append(n_ref)
        if n_ref == 0:
            edits.append(len(h_norm))
        else:
            out = jiwer.process_characters(r_norm, h_norm)
            edits.append(out.substitutions + out.deletions + out.insertions)

    edits_arr = np.array(edits, dtype=np.int64)
    lens_arr = np.array(ref_lens, dtype=np.int64)

    rng = np.random.default_rng(seed)
    n = len(references)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    sample_edits = edits_arr[idx].sum(axis=1)
    sample_lens = lens_arr[idx].sum(axis=1)
    valid = sample_lens > 0
    if not np.any(valid):
        return 0.0, 0.0

    sample_cers = sample_edits[valid] / sample_lens[valid]
    lower = float(np.percentile(sample_cers, 100 * (alpha / 2)))
    upper = float(np.percentile(sample_cers, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def stratify_by_duration(
    records: list[dict],
    references: list[str],
    hypotheses: list[str],
    bins: tuple[tuple[str, float, float], ...] = (
        ("short (<10s)", 0.0, 10.0),
        ("medium (10-20s)", 10.0, 20.0),
        ("long (>20s)", 20.0, float("inf")),
    ),
) -> dict[str, dict]:
    """Calculate clip count, WER, and CER grouped by clip duration."""
    stratified: dict[str, dict] = {}
    for label, min_dur, max_dur in bins:
        bin_refs: list[str] = []
        bin_hyps: list[str] = []
        for rec, ref, hyp in zip(records, references, hypotheses):
            dur = float(rec.get("duration_s", 0.0))
            if min_dur <= dur < max_dur:
                bin_refs.append(ref)
                bin_hyps.append(hyp)
        if bin_refs:
            stratified[label] = {
                "n_clips": len(bin_refs),
                "wer": round(compute_wer(bin_refs, bin_hyps), 4),
                "cer": round(compute_cer(bin_refs, bin_hyps), 4),
            }
        else:
            stratified[label] = {"n_clips": 0, "wer": None, "cer": None}
    return stratified
