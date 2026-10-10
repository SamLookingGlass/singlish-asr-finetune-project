from pathlib import Path

from src.config import PROJECT_ROOT, as_project_path, resolve_path


def test_project_relative_roundtrip_and_foreign_absolute_path():
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / "_path_test.wav"
    wav.write_bytes(b"not-a-real-wav")
    try:
        relative = as_project_path(wav)
        assert relative == "data/processed/audio/_path_test.wav"
        assert resolve_path(relative) == wav.resolve()

        foreign = (
            r"C:\Users\desktop\Desktop\AI Engineering Projects"
            r"\Project 2\data\processed\audio\_path_test.wav"
        )
        assert resolve_path(foreign) == wav.resolve()
    finally:
        wav.unlink(missing_ok=True)


def test_resolve_path_keeps_existing_file(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"x")
    assert resolve_path(wav) == wav
