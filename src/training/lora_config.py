"""LoRA settings: which Whisper weights get trainable low-rank adapters.

We attach adapters to the attention query and value projections, the
standard choice for Whisper. Everything else stays frozen.
"""

from __future__ import annotations

DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


def build_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
