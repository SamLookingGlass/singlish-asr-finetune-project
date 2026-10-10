"""Transcribe audio files with a local Whisper model, in batches.

Supports optional "contextual biasing": the hotword lexicon is turned into
a short prompt ("Singlish glossary: lah, shiok, kopitiam, ...") that is fed
to the model before decoding. Whisper treats the prompt as earlier
conversation context, which nudges it toward those words - the same idea
as sending a hotword list to a commercial ASR API.

Can also be run directly to write a hypotheses file (one JSON line per
clip), which hotword mining and evaluation both consume:

    python -m src.eval.decode --manifests data/manifests/train.jsonl \
        --variant base --out data/hyps/base_trainval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audio import load_mono
from src.config import settings

MAX_PROMPT_TERMS = 60  # Whisper's prompt window is 224 tokens; stay well under


def build_hotword_prompt(hotwords: list[str], max_terms: int = MAX_PROMPT_TERMS) -> str:
    """Join the top hotwords into one short prompt string."""
    return "Singlish glossary: " + ", ".join(hotwords[:max_terms]) + "."


class LocalWhisperTranscriber:
    """Transcribes audio with Whisper, with or without our LoRA adapter."""

    def __init__(
        self,
        model_id: str | None = None,
        adapter_path: str | None = None,
        hotwords: list[str] | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = model_id or settings.whisper_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.batch_size = batch_size

        self.processor = WhisperProcessor.from_pretrained(
            self.model_id, language="english", task="transcribe"
        )
        # Whisper is BPE. The WordPiece cleanup step strips spaces and warns.
        self.processor.tokenizer.clean_up_tokenization_spaces = False
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True
        )
        if adapter_path:
            from peft import PeftModel

            adapter_dir = Path(adapter_path).expanduser().resolve()
            config_file = adapter_dir / "adapter_config.json"
            if not config_file.is_file():
                raise FileNotFoundError(
                    f"LoRA adapter not found at {adapter_dir}\n"
                    f"Expected file: {config_file}\n"
                    "Re-run training. The adapter is the folder that contains adapter_config.json."
                )
            # Resolve to an absolute path so PEFT does not treat it as a Hub repo id.
            model = PeftModel.from_pretrained(
                model, str(adapter_dir), local_files_only=True
            )
            # Fold the LoRA weights into the base model so decoding runs at
            # normal speed, with no adapter overhead.
            model = model.merge_and_unload()
        self.model = model.to(self.device).eval()
        # Set language on the config. Passing language= into generate() as well
        # makes current transformers also send suppress_tokens, and it warns
        # that the two disagree.
        gen_cfg = self.model.generation_config
        gen_cfg.language = "english"
        gen_cfg.task = "transcribe"
        gen_cfg.max_new_tokens = 200
        gen_cfg.max_length = None

        self.prompt_ids = None
        if hotwords:
            prompt = build_hotword_prompt(hotwords)
            self.prompt_ids = self.processor.get_prompt_ids(
                prompt, return_tensors="pt"
            ).to(self.device)

    def _decode_ids(self, sequences) -> list[str]:
        """Turn generated token ids into text.

        When a prompt is used, the model's output starts with the prompt
        tokens. Everything before the <|startoftranscript|> marker is cut
        so the prompt text never leaks into the transcription.
        """
        sot_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        texts = []
        for seq in sequences:
            ids = seq.tolist()
            if sot_id in ids:
                ids = ids[ids.index(sot_id):]
            texts.append(
                self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            )
        return texts

    def transcribe(self, audio_paths: list[str]) -> list[str]:
        import torch
        from tqdm import tqdm

        hypotheses: list[str] = []
        for start in tqdm(
            range(0, len(audio_paths), self.batch_size), desc="decoding", unit="batch"
        ):
            batch_paths = audio_paths[start : start + self.batch_size]
            audios = [load_mono(path) for path in batch_paths]

            inputs = self.processor.feature_extractor(
                audios,
                sampling_rate=settings.sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self.device, dtype=self.dtype)
            attention_mask = inputs.attention_mask.to(self.device)

            generate_kwargs: dict = {"attention_mask": attention_mask}
            if self.prompt_ids is not None:
                generate_kwargs["prompt_ids"] = self.prompt_ids

            with torch.no_grad():
                generated = self.model.generate(input_features, **generate_kwargs)
            hypotheses.extend(self._decode_ids(generated))
        return hypotheses


def load_manifests(paths: list[str], limit: int | None = None) -> list[dict]:
    """Read one or more manifest JSONL files into a list of records."""
    records: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return records[:limit] if limit is not None else records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--variant", choices=["base", "finetuned"], default="base")
    parser.add_argument("--adapter-path", default=None, help="required for --variant finetuned")
    parser.add_argument("--hotwords-file", default=None, help="curated_lexicon.json for biasing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.variant == "finetuned" and not args.adapter_path:
        parser.error("--variant finetuned requires --adapter-path")

    hotwords = None
    if args.hotwords_file:
        with open(args.hotwords_file, encoding="utf-8") as f:
            lexicon = json.load(f)
        hotwords = [t["term"] for t in lexicon["terms"]]

    records = load_manifests(args.manifests, limit=args.limit)
    transcriber = LocalWhisperTranscriber(
        adapter_path=args.adapter_path, hotwords=hotwords, batch_size=args.batch_size
    )
    hypotheses = transcriber.transcribe([r["audio_path"] for r in records])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": rec["id"], "hypothesis": hyp}, ensure_ascii=False) + "\n" for rec, hyp in zip(records, hypotheses))
    print(f"Wrote {len(hypotheses)} hypotheses -> {out_path}")


if __name__ == "__main__":
    main()
