"""Commercial baseline: transcribe clips with OpenAI's gpt-4o-transcribe API.

Each clip is uploaded to the API one at a time, with retries on failure.
Every call costs money, so use ``limit`` to cap how many clips are sent.
"""

from __future__ import annotations

import time

from src.config import settings


def transcribe_openai(
    records: list[dict],
    model: str | None = None,
    limit: int | None = None,
    max_retries: int = 3,
) -> list[dict]:
    """Transcribe the given clips. Returns a list of {"id", "hypothesis"} dicts."""
    from openai import OpenAI
    from tqdm import tqdm

    client = OpenAI(api_key=settings.openai_api_key or None)
    model = model or settings.stt_baseline_model
    if limit is not None:
        records = records[:limit]

    results: list[dict] = []
    for rec in tqdm(records, desc=f"transcribing via {model}"):
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                with open(rec["audio_path"], "rb") as f:
                    response = client.audio.transcriptions.create(model=model, file=f)
                results.append({"id": rec["id"], "hypothesis": response.text})
                break
            except Exception as err:  # noqa: BLE001 - retry transient API failures
                last_err = err
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"transcription failed for {rec['id']}: {last_err}")
    return results
