"""Central configuration: project paths, plus settings read from the .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"


def as_project_path(path: Path | str) -> str:
    """Store a path relative to the repo when it lives inside the repo.

    Absolute paths from the machine that built a manifest break as soon as
    the project is cloned somewhere else.
    """
    p = Path(path)
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(p)
    return rel.as_posix()


def resolve_path(path: str | Path) -> Path:
    """Find a file named in a manifest on this machine.

    Tries the path as written, then relative to the repo root,
    then anchors from the ``data/`` directory if it was an absolute path from another machine.
    """
    p = Path(path)
    if p.is_file():
        return p.resolve()

    if not p.is_absolute():
        return (PROJECT_ROOT / p).resolve()

    # If an absolute path was serialized on another computer, re-anchor from data/
    parts = p.parts
    for i, part in enumerate(parts):
        if part.lower() == "data":
            candidate = (PROJECT_ROOT / Path(*parts[i:])).resolve()
            if candidate.is_file():
                return candidate
            return candidate

    return p.resolve()
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
