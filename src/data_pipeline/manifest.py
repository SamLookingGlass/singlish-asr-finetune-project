"""Split the cleaned corpus into train/val/test manifests, with size stats.

The split is deterministic: each clip's id is hashed into a bucket from
0-99, and the bucket decides the split (default 80/10/10). Re-running the
pipeline always gives the same split.

Known limitation (also noted in the README): SASRBench-v1 has no speaker
ids, so we cannot guarantee that a speaker in the training set never
appears in the test set. A clip itself only ever lands in one split.

Outputs:
- data/manifests/{train,val,test}.jsonl
- data/manifests/corpus_stats.json  (clip counts, hours, average length)
"""

from __future__ import annotations

import argparse
import hashlib
import json

from src.config import MANIFEST_DIR, PROCESSED_DIR
from src.eval.figures import write_corpus_figure

SPLITS = ("train", "val", "test")


def assign_split(clip_id: str, train_pct: int = 80, val_pct: int = 10) -> str:
    """Pick train, val, or test for a clip. Same id always gives the same answer."""
    bucket = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def split_stats(records: list[dict]) -> dict:
    """Count clips, total hours, and average clip length for one split."""
    if not records:
        return {"clips": 0, "hours": 0.0, "avg_clip_s": 0.0}
    total_s = sum(r["duration_s"] for r in records)
    return {
        "clips": len(records),
        "hours": round(total_s / 3600, 3),
        "avg_clip_s": round(total_s / len(records), 2),
    }


def build(train_pct: int = 80, val_pct: int = 10) -> dict:
    """Write the three split manifests and print a small stats table."""
    processed_manifest = PROCESSED_DIR / "processed_manifest.jsonl"
    with open(processed_manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for rec in records:
        by_split[assign_split(rec["id"], train_pct, val_pct)].append(rec)

    for split, recs in by_split.items():
        path = MANIFEST_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {split: split_stats(recs) for split, recs in by_split.items()}
    stats["total"] = split_stats(records)
    with open(MANIFEST_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"{'split':<8}{'clips':>8}{'hours':>10}{'avg s':>8}")
    for split in (*SPLITS, "total"):
        s = stats[split]
        print(f"{split:<8}{s['clips']:>8}{s['hours']:>10.2f}{s['avg_clip_s']:>8.2f}")
    figure = write_corpus_figure(stats)
    print(f"Wrote {figure}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pct", type=int, default=80)
    parser.add_argument("--val-pct", type=int, default=10)
    args = parser.parse_args()
    build(train_pct=args.train_pct, val_pct=args.val_pct)


if __name__ == "__main__":
    main()
