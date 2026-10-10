"""Clean up the raw audio in parallel: resample, trim silence, remove duplicates.

Each clip goes through the same three steps, one Ray task per clip so all
CPU cores are used. On Windows (where Ray has no wheels) the exact same
function runs on a normal Python process pool instead - only the scheduler
changes, not the logic.

Steps per clip:
1. Resample to 16 kHz mono (the rate Whisper expects).
2. Cut silence from the start and end using Silero VAD.
   Clips with no detected speech at all are dropped.
3. Hash the audio and the cleaned transcript, so duplicates can be found.

After all clips are processed, exact duplicates (same audio bytes, or same
transcript) are removed. The source recordings come from two people in the
same room, so the same speech can appear in both microphone channels.

Re-running skips clips that already have a processed WAV (resume-friendly).

Outputs:
- data/processed/audio/*.wav
- data/processed/processed_manifest.jsonl
- data/processed/preprocess_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from src.config import PROCESSED_DIR, RAW_DIR, as_project_path, resolve_path, settings
from src.eval.metrics import normalize_text

PROCESSED_MANIFEST = PROCESSED_DIR / "processed_manifest.jsonl"
STATS_PATH = PROCESSED_DIR / "preprocess_stats.json"

# The VAD model is loaded once per worker process, then reused.
_VAD_MODEL = None


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def _record_from_wav(record: dict, wav_path: Path) -> dict:
    """Build an ok record from an already-written processed WAV (resume path)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return {
        "id": record["id"],
        "audio_path": as_project_path(wav_path),
        "transcript": record["transcript"],
        "duration_s": round(len(data) / sr, 3),
        "sample_rate": sr,
        "audio_hash": hashlib.sha1(np.ascontiguousarray(data).tobytes()).hexdigest(),
        "text_hash": hashlib.sha1(normalize_text(record["transcript"]).encode()).hexdigest(),
        "status": "ok",
    }


def process_clip(record: dict, out_dir: str, target_sr: int, vad_pad_s: float = 0.1) -> dict:
    """Resample and silence-trim one clip.

    Returns the updated record with a "status" field:
    - "ok" if the clip was saved (or already existed)
    - "no_speech" if the VAD found no speech
    - "error" if something unexpected failed
    """
    try:
        out_path = Path(out_dir) / f"{record['id']}.wav"
        if out_path.exists():
            return _record_from_wav(record, out_path)

        import numpy as np
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps

        data, sr = sf.read(resolve_path(record["audio_path"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo -> mono
        if sr != target_sr:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        model = _get_vad_model()
        speech = get_speech_timestamps(torch.from_numpy(data), model, sampling_rate=sr)
        if not speech:
            return {**record, "status": "no_speech"}

        # Keep everything from the first to the last speech segment,
        # plus a small padding so words are not clipped.
        pad = int(vad_pad_s * sr)
        start = max(0, speech[0]["start"] - pad)
        end = min(len(data), speech[-1]["end"] + pad)
        trimmed = data[start:end]

        sf.write(out_path, trimmed, sr)

        return {
            "id": record["id"],
            "audio_path": as_project_path(out_path),
            "transcript": record["transcript"],
            "duration_s": round(len(trimmed) / sr, 3),
            "sample_rate": sr,
            "audio_hash": hashlib.sha1(np.ascontiguousarray(trimmed).tobytes()).hexdigest(),
            "text_hash": hashlib.sha1(normalize_text(record["transcript"]).encode()).hexdigest(),
            "status": "ok",
        }
    except Exception as err:  # noqa: BLE001 - keep the pool alive on bad clips
        return {**record, "status": "error", "error": str(err)}


def dedup(records: list[dict]) -> tuple[list[dict], dict]:
    """Remove clips whose audio or transcript was already seen. Keeps the first copy."""
    seen_audio: set[str] = set()
    seen_text: set[str] = set()
    kept: list[dict] = []
    n_dup_audio = n_dup_text = 0
    for rec in records:
        if rec["audio_hash"] in seen_audio:
            n_dup_audio += 1
            continue
        if rec["text_hash"] in seen_text:
            n_dup_text += 1
            continue
        seen_audio.add(rec["audio_hash"])
        seen_text.add(rec["text_hash"])
        kept.append(rec)
    return kept, {"n_dup_audio": n_dup_audio, "n_dup_text": n_dup_text}


def _map_with_ray(records: list[dict], audio_dir: str, num_cpus: int | None) -> list[dict]:
    """Run process_clip for every record as Ray tasks (Linux / Colab)."""
    import ray

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, include_dashboard=False)
    task = ray.remote(process_clip)
    futures = [task.remote(rec, audio_dir, settings.sample_rate) for rec in records]
    results = ray.get(futures)
    ray.shutdown()
    return results


def _map_with_process_pool(
    records: list[dict], audio_dir: str, num_cpus: int | None
) -> list[dict]:
    """Windows fallback: same work, standard-library process pool instead of Ray.

    Caps workers by default so each process loading Silero VAD does not OOM.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    workers = num_cpus or min(4, os.cpu_count() or 2)
    print(f"process-pool workers: {workers}")
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_clip, rec, audio_dir, settings.sample_rate): rec
            for rec in records
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocessing"):
            try:
                results.append(fut.result())
            except Exception as err:  # noqa: BLE001
                rec = futures[fut]
                results.append({**rec, "status": "error", "error": str(err)})
    return results


def run(num_cpus: int | None = None, limit: int | None = None) -> Path:
    """Preprocess every clip in the raw manifest and write the processed manifest."""
    raw_manifest = RAW_DIR / "raw_manifest.jsonl"
    with open(raw_manifest, encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f]
    if limit is not None:
        raw_records = raw_records[:limit]

    audio_dir = PROCESSED_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        import ray  # noqa: F401

        results = _map_with_ray(raw_records, str(audio_dir), num_cpus)
    except ImportError:
        print("ray not available on this platform - using process-pool fallback")
        results = _map_with_process_pool(raw_records, str(audio_dir), num_cpus)

    ok = [r for r in results if r["status"] == "ok"]
    n_no_speech = sum(1 for r in results if r["status"] == "no_speech")
    n_error = sum(1 for r in results if r["status"] == "error")
    if n_error:
        print(f"warning: {n_error} clips failed with errors (skipped)")
    kept, dup_stats = dedup(ok)

    with open(PROCESSED_MANIFEST, "w", encoding="utf-8") as f:
        for rec in kept:
            rec.pop("status", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "n_input": len(raw_records),
        "n_no_speech_dropped": n_no_speech,
        "n_error": n_error,
        **dup_stats,
        "n_kept": len(kept),
        "hours_input": round(sum(r["duration_s"] for r in raw_records) / 3600, 3),
        "hours_kept": round(sum(r["duration_s"] for r in kept) / 3600, 3),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Wrote {PROCESSED_MANIFEST}")
    return PROCESSED_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cpus", type=int, default=None, help="worker count (default: 4 on Windows)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N clips")
    args = parser.parse_args()
    run(num_cpus=args.num_cpus, limit=args.limit)


if __name__ == "__main__":
    main()
