from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ENAMEL CSV into open_llm_generation JSON list."
    )
    parser.add_argument(
        "--csv",
        default="enamel.csv",
        help="Path to ENAMEL csv file.",
    )
    parser.add_argument(
        "--output",
        default="enamel_openllm_input.json",
        help="Output JSON path for tools/open_llm_generation.py",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional prefix added before each prompt.",
    )
    parser.add_argument(
        "--subset",
        default="",
        help="Optional comma-separated problem ids; empty means all rows.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    subset = {int(x.strip()) for x in args.subset.split(",") if x.strip()}

    rows: list[dict] = []
    for idx, row in df.iterrows():
        pid = int(idx)
        if subset and pid not in subset:
            continue
        prompt = str(row["prompt"])
        if args.prompt_prefix:
            prompt = args.prompt_prefix + "\n\n" + prompt
        rows.append(
            {
                "problem_id": pid,
                "task_id": str(row["task_id"]),
                "entry_point": str(row["entry_point"]),
                "prompt": prompt,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
