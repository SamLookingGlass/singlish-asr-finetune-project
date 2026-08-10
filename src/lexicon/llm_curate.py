"""Filter the mined hotword candidates with an LLM (gpt-4o-mini).

Mining produces a noisy list: real Singlish terms mixed with ordinary
English words the model happened to mishear. This step asks the LLM to
label each candidate as "hotword" (a real Singlish/Malay/Hokkien term) or
"noise" (ordinary English or an ASR artifact), using one of the prompt
versions in prompts.py.

The accepted terms become the curated lexicon, which is used for
contextual biasing during decoding and for the term-recall metric.

Outputs:
- data/lexicon/curated_lexicon.json    (accepted terms)
- data/lexicon/curation_details.json   (every LLM decision, for inspection)
"""

from __future__ import annotations

import argparse
import json
import time

from src.config import LEXICON_DIR, settings
from src.lexicon.prompts import PROMPT_VERSIONS, SYSTEM_PROMPT

CURATED_LEXICON = LEXICON_DIR / "curated_lexicon.json"
CURATION_DETAILS = LEXICON_DIR / "curation_details.json"


def get_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key or None)


def classify_term(
    client,
    term: str,
    examples: list[dict] | None = None,
    prompt_version: str = "v2",
    model: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Ask the LLM to label one term. Retries with backoff if the API call fails.

    Returns a dict with the label, category, reason, latency, and token counts.
    """
    template = PROMPT_VERSIONS[prompt_version]
    example_str = (
        "; ".join(f"ref: '{e['ref']}' -> asr heard: '{e['hyp'] or '(deleted)'}'" for e in examples)
        if examples
        else "(none)"
    )
    prompt = template.format(term=term, examples=example_str)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model or settings.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            latency = time.perf_counter() - t0
            parsed = json.loads(response.choices[0].message.content)
            label = str(parsed.get("label", "noise")).lower()
            return {
                "term": term,
                "label": label if label in ("hotword", "noise") else "noise",
                "category": parsed.get("category", ""),
                "reason": parsed.get("reason", ""),
                "prompt_version": prompt_version,
                "latency_s": round(latency, 3),
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }
        except Exception as err:  # noqa: BLE001 - retry any API/parse failure
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"classification failed for term '{term}': {last_err}")


def curate(
    candidates: list[dict],
    prompt_version: str = "v2",
    model: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Label every candidate. Returns (accepted hotwords, all decisions)."""
    client = get_client()
    if limit is not None:
        candidates = candidates[:limit]

    decisions: list[dict] = []
    for i, cand in enumerate(candidates):
        decision = classify_term(
            client,
            cand["term"],
            examples=cand.get("examples"),
            prompt_version=prompt_version,
            model=model,
        )
        decision["mined_count"] = cand.get("count", 0)
        decisions.append(decision)
        if (i + 1) % 25 == 0:
            print(f"  classified {i + 1}/{len(candidates)}")

    accepted = [d for d in decisions if d["label"] == "hotword"]
    return accepted, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(LEXICON_DIR / "mined_candidates.json"))
    parser.add_argument("--prompt-version", choices=sorted(PROMPT_VERSIONS), default="v2")
    parser.add_argument("--model", default=None, help="override LLM model")
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N")
    args = parser.parse_args()

    with open(args.candidates, encoding="utf-8") as f:
        candidates = json.load(f)

    accepted, decisions = curate(
        candidates, prompt_version=args.prompt_version, model=args.model, limit=args.limit
    )

    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    with open(CURATED_LEXICON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_version": args.prompt_version,
                "model": args.model or settings.llm_model,
                "terms": [
                    {"term": d["term"], "category": d["category"], "mined_count": d["mined_count"]}
                    for d in accepted
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(CURATION_DETAILS, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    print(
        f"Accepted {len(accepted)}/{len(decisions)} candidates as hotwords -> {CURATED_LEXICON}"
    )


if __name__ == "__main__":
    main()
