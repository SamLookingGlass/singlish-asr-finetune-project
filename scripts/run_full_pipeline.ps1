# End-to-end local pipeline. Training uses the GPU when PyTorch can see one.
#
# Usage:
#   .\scripts\run_full_pipeline.ps1
#   .\scripts\run_full_pipeline.ps1 -Smoke
#
# -Smoke still downloads the corpus, then trains and evaluates on a small slice.

param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv yet. Run .\scripts\setup.ps1 first."
    exit 1
}

$slice = @()
if ($Smoke) {
    $slice = @("--limit", "32")
}

Write-Host "[1/5] Downloading SASRBench-v1..."
& $python -m src.data_pipeline.download @slice
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/5] Preprocessing (resample + VAD trim + dedup)..."
& $python -m src.data_pipeline.ray_preprocess @slice
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/5] Building train/val/test manifests..."
& $python -m src.data_pipeline.manifest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/5] LoRA fine-tune..."
$trainArgs = @(
    "-m", "src.training.train",
    "--output-dir", "checkpoints/whisper-turbo-singlish-lora"
)
if ($Smoke) {
    $trainArgs += @(
        "--limit", "32",
        "--max-steps", "20",
        "--batch-size", "2",
        "--grad-accum", "1",
        "--eval-steps", "10",
        "--warmup-steps", "0"
    )
} else {
    $trainArgs += @("--epochs", "3", "--batch-size", "8", "--grad-accum", "2")
}
& $python @trainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/5] Evaluating base Whisper and the LoRA adapter..."
$evalArgs = @("-m", "src.eval.run_eval")
if ($Smoke) {
    $evalArgs += @("--limit", "8")
}
& $python @evalArgs --variant base
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python @evalArgs --variant finetuned --adapter-path checkpoints/whisper-turbo-singlish-lora
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m src.eval.run_eval --compare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Comparison table: data/eval/comparison.md"
Write-Host "Hotword mining is separate (needs a train+val decode and an OpenAI key):"
Write-Host "  python -m src.eval.decode --manifests data/manifests/train.jsonl data/manifests/val.jsonl --variant base --out data/hyps/base_trainval.jsonl"
Write-Host "  python -m src.lexicon.mine_hotwords --hyps data/hyps/base_trainval.jsonl"
