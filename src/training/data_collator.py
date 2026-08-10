"""Batch collator for Whisper training: pads audio features and labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Turns a list of examples into one padded training batch.

    Audio features are padded to Whisper's fixed 30-second window, and label
    token ids are padded to the longest label in the batch. Padded label
    positions are set to -100 so the loss ignores them.

    If the tokenizer already added a BOS token to every label, it is removed
    here, because the trainer adds its own decoder start token.
    """

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
