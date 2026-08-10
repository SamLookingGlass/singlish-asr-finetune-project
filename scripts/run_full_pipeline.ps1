# End-to-end local pipeline (CPU). Training itself runs on Colab - see
# notebooks/finetune_whisper_lora_colab.ipynb.
#
# Usage:  .\scripts\run_full_pipeline.ps1
$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python.exe"

Write-Host "[1/5] Downloading SASRBench-v1..."
& $python -m src.data_pipeline.download

Write-Host "[2/5] Preprocessing (resample + VAD trim + dedup)..."
& $python -m src.data_pipeline.ray_preprocess

Write-Host "[3/5] Building train/val/test manifests..."
& $python -m src.data_pipeline.manifest

Write-Host "[4/5] Baseline decode of train+val (CPU - this is slow; consider running on Colab)..."
& $python -m src.eval.decode --manifests data/manifests/train.jsonl data/manifests/val.jsonl --variant base --out data/hyps/base_trainval.jsonl

Write-Host "[5/5] Mining hotword candidates + LLM curation..."
& $python -m src.lexicon.mine_hotwords --hyps data/hyps/base_trainval.jsonl
& $python -m src.lexicon.llm_curate

Write-Host "Done. Next: run the Colab notebook to fine-tune, then 'python -m src.eval.run_eval --compare'."
