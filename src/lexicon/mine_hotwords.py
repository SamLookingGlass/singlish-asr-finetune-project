"""Find Singlish terms the baseline ASR model keeps getting wrong.

The idea is simple: compare what the speaker actually said (the reference
transcript) with what the model heard (the hypothesis). Words that the
model often replaces or drops are good candidates for the hotword lexicon -
usually Singlish particles, Malay/Hokkien loanwords, and local names.
This is the same mining approach teams use on production ASR logs.

Run order: first produce baseline hypotheses with ``src.eval.decode``,
then run this script, then curate the candidates with ``src.lexicon.llm_curate``.

How it works:
1. Align each reference/hypothesis pair word by word (using jiwer).
2. Collect the reference words involved in substitutions and deletions.
3. Drop common English words and rare one-off errors, rank the rest by count.
4. Always include the seed Singlish terms that appear in the corpus, even
   if the model got them right - the lexicon should still cover them.

Output: data/lexicon/mined_candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from src.config import LEXICON_DIR, MANIFEST_DIR

MINED_CANDIDATES = LEXICON_DIR / "mined_candidates.json"

# A small list of well-known Singlish particles and Malay/Hokkien loanwords.
# It only bootstraps the process - the point is to *mine* the rest.
SEED_TERMS = {
    "lah", "lor", "leh", "meh", "hor", "sia", "sian", "shiok", "kiasu",
    "makan", "paiseh", "alamak", "aiyah", "aiyo", "walao", "atas", "kaypoh",
    "chope", "lepak", "tapao", "kopitiam", "ang moh", "bojio", "jialat",
}

# Common English words that show up as ASR errors but are never domain
# terms. Kept small on purpose - the LLM curation step does the fine
# filtering afterwards.
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "there", "here", "is", "am", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "doing", "have",
    "has", "had", "having", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "of", "to", "in", "on", "at", "by",
    "for", "with", "about", "from", "into", "over", "under", "up", "down",
    "out", "off", "so", "as", "not", "no", "yes", "yeah", "okay", "ok",
    "oh", "ah", "um", "uh", "er", "hmm", "like", "just", "very", "really",
    "quite", "some", "any", "all", "one", "two", "three", "what", "when",
    "where", "which", "who", "whom", "why", "how", "because", "also",
    "too", "only", "even", "still", "again", "go", "going", "went", "get",
    "got", "know", "think", "want", "say", "said", "see", "come", "came",
    "dont", "don't", "cant", "can't", "im", "i'm", "it's", "actually",
    "right", "now", "more", "most", "much", "many", "other", "thing",
    "things", "people", "time", "good", "bad", "make", "made", "take", "took",
}


def normalize_word(word: str) -> str:
    """Lowercase a single word and strip punctuation (keeps apostrophes)."""
    return re.sub(r"[^\w']", "", word.lower()).strip("_")


def _tokenize(text: str) -> list[str]:
    return [w for w in (normalize_word(t) for t in text.split()) if w]


def _collect_errors(pairs, sub_counts, del_counts, examples, ref_vocab, max_examples):
    """Word-align each pair and count which reference words were misheard."""
    import jiwer

    for ref, hyp in pairs:
        ref_tokens = _tokenize(ref)
        hyp_tokens = _tokenize(hyp)
        ref_vocab.update(ref_tokens)
        if not ref_tokens:
            continue

        if not hyp_tokens:
            # The model produced nothing: every reference word is a deletion.
            for word in ref_tokens:
                del_counts[word] += 1
            continue

        out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
        for chunk in out.alignments[0]:
            if chunk.type not in ("substitute", "delete"):
                continue
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                word = ref_tokens[idx]
                if chunk.type == "substitute":
                    sub_counts[word] += 1
                else:
                    del_counts[word] += 1
                if len(examples[word]) < max_examples:
                    hyp_span = (
                        " ".join(hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])
                        if chunk.type == "substitute"
                        else ""
                    )
                    examples[word].append(
                        {
                            "ref": " ".join(ref_tokens[max(0, idx - 3) : idx + 4]),
                            "hyp": hyp_span,
                            "error": chunk.type,
                        }
                    )


def mine_candidates(
    pairs: list[tuple[str, str]],
    min_count: int = 2,
    min_len: int = 2,
    max_examples: int = 3,
) -> list[dict]:
    """Mine candidate hotwords from (reference, hypothesis) text pairs.

    Returns candidates sorted by how often they were misheard, each with
    error counts and up to ``max_examples`` example snippets.
    """
    sub_counts: Counter[str] = Counter()
    del_counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    ref_vocab: Counter[str] = Counter()

    _collect_errors(pairs, sub_counts, del_counts, examples, ref_vocab, max_examples)

    candidates: list[dict] = []
    error_words = set(sub_counts) | set(del_counts)
    for word in error_words:
        total = sub_counts[word] + del_counts[word]
        is_seed = word in SEED_TERMS
        # Seed terms always pass; other words must be frequent enough,
        # long enough, and not a common English word.
        if not is_seed and (
            total < min_count or len(word) < min_len or word in ENGLISH_STOPWORDS
        ):
            continue
        candidates.append(
            {
                "term": word,
                "count": total,
                "sub_count": sub_counts[word],
                "del_count": del_counts[word],
                "seed": is_seed,
                "examples": examples[word],
            }
        )

    # Seed terms the model got right still join the pool (with count 0),
    # so the final lexicon covers them too.
    for term in SEED_TERMS:
        if term in ref_vocab and term not in error_words:
            candidates.append(
                {
                    "term": term,
                    "count": 0,
                    "sub_count": 0,
                    "del_count": 0,
                    "seed": True,
                    "examples": [],
                }
            )

    candidates.sort(key=lambda c: (-c["count"], c["term"]))
    return candidates


def load_pairs(manifest_paths: list[str], hyps_path: str) -> list[tuple[str, str]]:
    """Match manifest transcripts with decoded hypotheses using the clip id."""
    hyps: dict[str, str] = {}
    with open(hyps_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            hyps[rec["id"]] = rec["hypothesis"]

    pairs: list[tuple[str, str]] = []
    missing = 0
    for manifest_path in manifest_paths:
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] in hyps:
                    pairs.append((rec["transcript"], hyps[rec["id"]]))
                else:
                    missing += 1
    if missing:
        print(f"warning: {missing} manifest clips had no hypothesis, skipped")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=[str(MANIFEST_DIR / "train.jsonl"), str(MANIFEST_DIR / "val.jsonl")],
    )
    parser.add_argument("--hyps", required=True, help="baseline hypotheses JSONL from decode.py")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--out", default=str(MINED_CANDIDATES))
    args = parser.parse_args()

    pairs = load_pairs(args.manifests, args.hyps)
    candidates = mine_candidates(pairs, min_count=args.min_count)

    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"Mined {len(candidates)} candidates from {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
