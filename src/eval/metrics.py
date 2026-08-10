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
