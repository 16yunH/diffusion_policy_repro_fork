#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_expected_score(checkpoint: Path) -> Optional[float]:
    match = re.search(r"test_mean_score=([0-9]+(?:\.[0-9]+)?)", checkpoint.name)
    if match is None:
        return None
    return float(match.group(1))


def find_score(eval_log: dict) -> Optional[float]:
    value = eval_log.get("test/mean_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--env-info", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--tolerance", default=0.05, type=float)
    args = parser.parse_args()

    eval_log_path = args.output_dir / "eval_log.json"
    if not eval_log_path.exists():
        raise FileNotFoundError(f"Missing eval log: {eval_log_path}")

    eval_log = load_json(eval_log_path)
    score = find_score(eval_log)
    expected = parse_expected_score(args.checkpoint)
    videos = sorted((args.output_dir / "media").glob("*.mp4"))
    status = "unknown"
    if score is not None and expected is not None:
        status = "pass" if abs(score - expected) <= args.tolerance else "check"
    elif score is not None:
        status = "score_present"

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Push-T Image Pretrained Eval Summary",
        "",
        f"- output_dir: `{args.output_dir}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- eval_log: `{eval_log_path}`",
        f"- test/mean_score: `{score}`",
        f"- expected_from_checkpoint_name: `{expected}`",
        f"- comparison_tolerance: `{args.tolerance}`",
        f"- status: `{status}`",
        f"- video_count: `{len(videos)}`",
        f"- env_info: `{args.env_info}`",
        "",
        "## Videos",
    ]
    if videos:
        lines.extend(f"- `{video}`" for video in videos)
    else:
        lines.append("- none")
    lines.extend(["", "## Eval Log Keys"])
    lines.extend(f"- `{key}`" for key in sorted(eval_log.keys()))
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "score": score,
        "expected": expected,
        "status": status,
        "video_count": len(videos),
        "summary_md": str(args.summary_md),
    }, indent=2))


if __name__ == "__main__":
    main()
