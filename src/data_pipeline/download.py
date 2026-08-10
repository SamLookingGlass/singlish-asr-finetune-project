"""Download the SASRBench-v1 corpus from Hugging Face into data/raw/.

SASRBench-v1 is a public set of 3,747 Singlish audio clips with
transcripts, taken from the IMDA National Speech Corpus (Part 3,
close-talk microphone). It ships as one "test" split.

Each clip is saved as a WAV file, and a manifest file (one JSON line per
clip) records the path, transcript, and duration. Later pipeline steps
only read these files.

We decode the audio bytes ourselves with soundfile instead of letting the
datasets library do it. This sidesteps backend differences between
datasets versions (newer ones use torchcodec, which needs FFmpeg on
Windows).
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from src.config import RAW_DIR, settings

DATASET_ID = "mjwong/SASRBench-v1"
RAW_MANIFEST = RAW_DIR / "raw_manifest.jsonl"


def download(limit: int | None = None) -> Path:
    """Download all clips, save them as WAVs, and write the raw manifest."""
    from datasets import Audio, load_dataset

    # huggingface_hub reads HF_TOKEN from the environment.
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    audio_dir = RAW_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    token = settings.hf_token or None
    ds = load_dataset(DATASET_ID, split="test", token=token)
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    records = []
    for i, row in enumerate(tqdm(ds, desc="downloading clips")):
        clip_id = f"sasr_{i:05d}"
        raw_audio = row["audio"]
        data, sr = sf.read(io.BytesIO(raw_audio["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo -> mono
        out_path = audio_dir / f"{clip_id}.wav"
        sf.write(out_path, data, sr)
        records.append(
            {
                "id": clip_id,
                "audio_path": str(out_path),
                "transcript": row["transcript"],
                "duration_s": round(len(data) / sr, 3),
                "sample_rate": sr,
            }
        )

    with open(RAW_MANIFEST, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)

    total_hours = sum(r["duration_s"] for r in records) / 3600
    print(f"Downloaded {len(records)} clips ({total_hours:.2f} h) -> {RAW_MANIFEST}")
    return RAW_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only download the first N clips")
    args = parser.parse_args()
    download(limit=args.limit)


if __name__ == "__main__":
    main()
