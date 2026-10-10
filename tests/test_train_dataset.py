import wave

from src.config import PROJECT_ROOT
from src.training.train import ManifestSpeechDataset


class _Features:
    def __init__(self, values):
        self.input_features = [values]


class _Ids:
    def __init__(self, values):
        self.input_ids = values


class _FakeExtractor:
    def __call__(self, audio, sampling_rate):
        return _Features([float(sampling_rate), float(len(audio))])


class _FakeTokenizer:
    def __call__(self, text):
        return _Ids([len(text)])


class _FakeProcessor:
    feature_extractor = _FakeExtractor()
    tokenizer = _FakeTokenizer()


def test_dataset_reads_one_clip_without_caching_features():
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / "_dataset_test.wav"
    with wave.open(str(wav), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 1600)
    try:
        ds = ManifestSpeechDataset(
            [{"audio_path": "data/processed/audio/_dataset_test.wav", "transcript": "lah"}],
            _FakeProcessor(),
        )
        item = ds[0]
        assert item["labels"] == [3]
        assert item["input_features"][0] == 16_000
        assert not hasattr(ds, "map")
    finally:
        wav.unlink(missing_ok=True)


def test_corrupted_or_missing_audio_raises_error():
    import pytest

    # Missing file
    ds_missing = ManifestSpeechDataset(
        [{"audio_path": "data/processed/audio/_non_existent.wav", "transcript": "test"}],
        _FakeProcessor(),
    )
    with pytest.raises(FileNotFoundError):
        _ = ds_missing[0]

    # Corrupted / invalid file
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    corrupted_wav = audio_dir / "_corrupted_test.wav"
    corrupted_wav.write_bytes(b"NOT_A_VALID_RIFF_HEADER_1234567890")
    try:
        ds_corrupt = ManifestSpeechDataset(
            [{"audio_path": "data/processed/audio/_corrupted_test.wav", "transcript": "test"}],
            _FakeProcessor(),
        )
        with pytest.raises(RuntimeError) as exc_info:
            _ = ds_corrupt[0]
        assert "Corrupted or unreadable audio file" in str(exc_info.value)
    finally:
        corrupted_wav.unlink(missing_ok=True)
