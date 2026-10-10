"""Fine-tune whisper-large-v3-turbo on the Singlish manifests using LoRA.

LoRA trains a small set of extra weights instead of the whole model.
The local notebook (notebooks/finetune_whisper_lora.ipynb) calls this script.

Clips are read during training, one batch at a time. An earlier version
precomputed every mel spectrogram with datasets.map. That still spawned a
subprocess when num_proc was 1, and on Colab the child was killed once the
feature cache got large, so training never started.

Precision is picked automatically: bf16 on newer GPUs, fp16 on a T4,
fp32 on CPU. Only the LoRA adapter (a few MB) is saved, not the full model.

Example:
    python -m src.training.train \
        --train-manifest data/manifests/train.jsonl \
        --val-manifest data/manifests/val.jsonl \
        --output-dir checkpoints/whisper-turbo-singlish-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.audio import load_mono
from src.config import MANIFEST_DIR, settings
from src.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.lora_config import build_lora_config


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records[:limit] if limit is not None else records


class ManifestSpeechDataset:
    """One manifest row at a time: audio features plus label token ids.

    A plain sequence (``__len__`` / ``__getitem__``) is enough for the
    Trainer. Features are not written into an Arrow cache first.
    """

    def __init__(self, records: list[dict], processor):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        audio = load_mono(rec["audio_path"])
        features = self.processor.feature_extractor(
            audio, sampling_rate=settings.sample_rate
        ).input_features[0]
        labels = self.processor.tokenizer(rec["transcript"]).input_ids
        return {"input_features": features, "labels": labels}


def select_precision() -> dict[str, bool]:
    """Pick bf16 (modern GPU), fp16 (older GPU like T4), or fp32 (CPU)."""
    import torch

    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


def train(args: argparse.Namespace) -> Path:
    import torch
    from peft import get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(
        args.model_id, language="english", task="transcribe"
    )

    print("Preparing datasets (audio is loaded per batch, not cached up front)...")
    train_ds = ManifestSpeechDataset(
        load_manifest(args.train_manifest, args.limit), processor
    )
    val_limit = None if args.limit is None else max(1, args.limit // 4)
    val_ds = ManifestSpeechDataset(load_manifest(args.val_manifest, val_limit), processor)
    print(f"train: {len(train_ds)} clips, val: {len(val_ds)} clips")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    # Don't force language/task tokens during training - the model learns
    # them from the labels instead.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False  # caching conflicts with gradient checkpointing

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # The base model is frozen, so without this hook no gradient would
        # flow back through the encoder to reach the LoRA weights.
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, build_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha))
    model.print_trainable_parameters()

    precision = select_precision()
    print(f"precision: {precision}, device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=25,
        save_total_limit=2,
        report_to=["none"],
        # Needed with PEFT: the wrapper hides the model's argument names,
        # so the trainer must not drop columns or guess the label name.
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_workers,
        **precision,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    model.save_pretrained(out_dir)  # adapter weights only
    processor.save_pretrained(out_dir)
    print(f"Saved LoRA adapter + processor to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=str(MANIFEST_DIR / "train.jsonl"))
    parser.add_argument("--val-manifest", default=str(MANIFEST_DIR / "val.jsonl"))
    parser.add_argument("--model-id", default=settings.whisper_model)
    parser.add_argument("--output-dir", default="checkpoints/whisper-turbo-singlish-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0, help="overrides epochs when > 0")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N train clips (and about N/4 val clips)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0 if sys.platform == "win32" else 2,
        help="DataLoader workers. Default 0 on Windows (spawn + processor pickling).",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
