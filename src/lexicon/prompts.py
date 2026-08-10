"""The two prompt versions used to classify hotword candidates.

v1 is a plain zero-shot prompt; v2 adds decision rules and examples.
The regression CI compares both versions against the hand-labeled gold set
(data/lexicon/gold_labels.json) - editing this file triggers that workflow,
and a merge is blocked if the newer prompt scores worse.

Both prompts must make the model return one JSON object:
    {"label": "hotword" | "noise", "category": "<short category>", "reason": "<one sentence>"}
"""

SYSTEM_PROMPT = (
    "You are a linguistics annotator helping build a contextual-biasing lexicon "
    "for a Singlish (Singapore English) speech recognition system. "
    "Always answer with a single JSON object and nothing else."
)

# v1: zero-shot, minimal instructions. Baseline prompt.
PROMPT_V1 = """\
Classify the candidate term below.

A term is a "hotword" if it is a genuine Singlish particle, a Malay, Hokkien,
Cantonese or Tamil loanword used in Singapore English, a Singapore-specific
name (place, institution, food), or a local abbreviation. Otherwise it is
"noise" (ordinary English, filler, or an ASR artifact).

Candidate term: "{term}"
Example usage from corpus (may be empty): {examples}

Return JSON: {{"label": "hotword" or "noise", "category": "<short category>", "reason": "<one sentence>"}}
"""

# v2: few-shot with explicit decision rules and hard negative examples.
PROMPT_V2 = """\
You will classify a candidate term mined from ASR errors on Singlish speech.

Decision rules, apply in order:
1. Singlish discourse particles (lah, lor, leh, meh, hor, sia...) -> hotword
2. Malay / Hokkien / Cantonese / Tamil loanwords used in Singapore English
   (makan, paiseh, kiasu, tahan, sedap...) -> hotword
3. Singapore-specific proper nouns and local terms: places (Tampines,
   Yishun), institutions and abbreviations (HDB, CPF, MRT, NTUC), local food
   (laksa, char kway teow) -> hotword
4. Ordinary English words, casual contractions (gonna, dunno), hesitation
   fillers (um, erm), and truncated ASR fragments (becau, singapo) -> noise
5. Ambiguous generic interjections that appear in all English varieties
   (eh, ya, oh) -> noise

Examples:
- "shiok" -> {{"label": "hotword", "category": "loanword", "reason": "Malay-origin slang for excellent, unique to Singapore English."}}
- "chope" -> {{"label": "hotword", "category": "singlish_verb", "reason": "Singlish verb meaning to reserve a seat."}}
- "actually" -> {{"label": "noise", "category": "english_filler", "reason": "Ordinary English discourse marker, not Singapore-specific."}}
- "becau" -> {{"label": "noise", "category": "asr_fragment", "reason": "Truncated fragment of 'because', an ASR artifact."}}
- "eh" -> {{"label": "noise", "category": "generic_interjection", "reason": "Generic interjection found in all English varieties."}}

Candidate term: "{term}"
Example usage from corpus (may be empty): {examples}

Return JSON: {{"label": "hotword" or "noise", "category": "<short category>", "reason": "<one sentence>"}}
"""

PROMPT_VERSIONS: dict[str, str] = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}
