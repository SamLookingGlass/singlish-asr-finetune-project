"""Central configuration: project paths, plus settings read from the .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
LEXICON_DIR = DATA_DIR / "lexicon"
EVAL_DIR = DATA_DIR / "eval"
HYPS_DIR = DATA_DIR / "hyps"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    hf_token: str = ""  # Hugging Face Hub token (faster / more reliable downloads)

    # Models
    llm_model: str = "gpt-4o-mini"
    stt_baseline_model: str = "gpt-4o-transcribe"
    whisper_model: str = "openai/whisper-large-v3-turbo"

    # Audio
    sample_rate: int = 16_000

    # The CI gate fails when the candidate prompt's F1 drops more than
    # this many points below the baseline (0.05 = 5 F1 points).
    regression_threshold_f1: float = 0.05


settings = Settings()
