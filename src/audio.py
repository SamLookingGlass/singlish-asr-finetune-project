"""Read a clip as mono float32 at the project sample rate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import resolve_path, settings


def load_mono(path: str | Path) -> np.ndarray:
    """Load one wav, downmix to mono, and resample to ``settings.sample_rate``."""
    import soundfile as sf

    audio_path = resolve_path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio not found: {path}\n"
            f"Resolved to: {audio_path}\n"
            "Re-run download, preprocess, and manifest on this machine. "
            "A manifest copied from another computer will not have the wavs."
        )
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception as err:
        raise RuntimeError(
            f"Corrupted or unreadable audio file at {audio_path}: {err}"
        ) from err

    if len(audio) == 0:
        raise ValueError(f"Empty audio file (0 samples): {audio_path}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != settings.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)
