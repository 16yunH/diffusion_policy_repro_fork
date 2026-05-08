#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json_lines(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    rows = []
    for seed_dir in sorted(args.run_root.glob("seed_*")):
        logs = read_json_lines(seed_dir / "logs.json.txt")
        best = None
        for item in logs:
            score = item.get("test/mean_score")
            if isinstance(score, (int, float)):
                if best is None or score > best["score"]:
                    best = {
                        "score": float(score),
                        "epoch": item.get("epoch"),
                        "global_step": item.get("global_step"),
                    }
        ckpts = sorted((seed_dir / "checkpoints").glob("*.ckpt"))
        rows.append({
            "run": seed_dir.name,
            "best": best,
            "checkpoint_count": len(ckpts),
            "latest_checkpoint": str(ckpts[-1]) if ckpts else None,
            "log_exists": (seed_dir / "train.log").exists(),
        })

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
