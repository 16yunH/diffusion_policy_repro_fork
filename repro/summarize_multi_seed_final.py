#!/usr/bin/env python3
"""Summarize multi-seed Push-T Image UNet Hybrid evaluation results."""
import argparse
import json
import re
from pathlib import Path
from typing import Optional


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_score(eval_log: dict) -> Optional[float]:
    value = eval_log.get("test/mean_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    parser.add_argument("--output", default="data/repro/multi_seed_final_summary.md")
    parser.add_argument("--tolerance", default=0.05, type=float)
    args = parser.parse_args()

    ref_score = 0.884  # Official Push-T Image UNet Hybrid score

    rows = []
    for eval_dir in sorted(args.eval_dirs):
        seed_match = re.search(r"seed(\d+)", eval_dir.name)
        seed = int(seed_match.group(1)) if seed_match else None

        eval_log_path = eval_dir / "eval_log.json"
        if not eval_log_path.exists():
            rows.append({"seed": seed, "error": f"Missing: {eval_log_path}"})
            continue

        eval_log = load_json(eval_log_path)
        score = find_score(eval_log)

        ckpt_glob = sorted(eval_dir.parent.parent.glob(f"*{eval_dir.name}*/**/*.ckpt"), key=lambda p: p.name)
        best_ckpt = None
        for ckpt in sorted((eval_dir.parent.parent / ".." / ".." / "outputs").rglob(f"**/seed_{seed}/**/*.ckpt")):
            pass  # We'll look at the checkpoint name from eval_dir itself

        videos = sorted((eval_dir / "media").glob("*.mp4")) if (eval_dir / "media").exists() else []

        status = "unknown"
        if score is not None:
            if abs(score - ref_score) >= args.tolerance:
                status = "pass" if score > ref_score else "below"
            else:
                status = "pass"

        rows.append({
            "seed": seed,
            "score": score,
            "expected": ref_score,
            "status": status,
            "video_count": len(videos),
            "eval_log": str(eval_log_path),
        })

    # Print JSON summary
    print(json.dumps(rows, indent=2, default=str))

    # Write markdown
    lines = [
        "# Push-T Image UNet Hybrid — Multi-Seed Final Results",
        "",
        f"Reference (official): `{ref_score}`",
        "",
        "| Seed | test/mean_score | Status | Videos |",
        "|------|-----------------|--------|--------|",
    ]
    scores = []
    for r in rows:
        score_str = f"{r['score']:.4f}" if r.get("score") else "N/A"
        lines.append(f"| {r['seed']} | {score_str} | {r['status']} | {r['video_count']} |")
        if r.get("score"):
            scores.append(r["score"])

    if scores:
        avg = sum(scores) / len(scores)
        lines.append("")
        lines.append(f"**Average**: {avg:.4f}  ")
        lines.append(f"**Std**: {std(scores):.4f}" if len(scores) > 1 else "")

    lines.append("")
    for r in rows:
        lines.append(f"## Seed {r['seed']}")
        lines.append(f"- eval_log: `{r.get('eval_log')}`")
        lines.append(f"- score: `{r.get('score')}`")
        lines.append(f"- status: `{r.get('status')}`")
        lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote: {output_path}")


def std(vals):
    import math
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


if __name__ == "__main__":
    main()
