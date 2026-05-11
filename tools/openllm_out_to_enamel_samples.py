from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path


IMPORT_PKG = """from typing import *
from bisect import *
from collections import *
from copy import *
from heapq import *
from math import *
from itertools import *
from functools import *
import string, re, math, random, itertools, functools
"""


def extract_code_block(text: str) -> str:
    matches = re.findall(r"```python\\s*(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```\\s*(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def normalize_solution(prompt: str, entry_point: str, completion: str) -> str:
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    code = extract_code_block(completion)
    prompt = prompt.rstrip("\n")
    target_def = f"def {entry_point}("

    if target_def in code:
        normalized = code[code.rfind(target_def) :].strip()
    else:
        body = textwrap.dedent(code).strip("\n")
        indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
        normalized = prompt if not indented else (prompt + "\n" + indented)

    full = IMPORT_PKG + "\n" + normalized.strip() + "\n"
    try:
        ast.parse(full)
    except SyntaxError:
        full = IMPORT_PKG + "\n" + prompt.strip() + "\n"
    return full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert open_llm_generation output JSON to ENAMEL samples format."
    )
    parser.add_argument(
        "--input",
        default="enamel_openllm_output.json",
        help="Input JSON path produced by tools/open_llm_generation.py",
    )
    parser.add_argument(
        "--output",
        default="l4e-qwen25-7b-instruct-enamel.json",
        help="Output ENAMEL samples json path.",
    )
    parser.add_argument(
        "--completion-key",
        default="completion",
        help="Field containing model output text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = json.loads(input_path.read_text(encoding="utf-8"))

    samples: dict[str, list[str]] = {}
    missing_completion = 0

    for item in rows:
        pid = int(item["problem_id"])
        prompt = str(item["prompt"])
        entry_point = str(item["entry_point"])
        completion = str(item.get(args.completion_key, ""))
        if not completion:
            missing_completion += 1
        code = normalize_solution(prompt=prompt, entry_point=entry_point, completion=completion)
        samples[str(pid)] = [code]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} tasks -> {out}")
    if missing_completion:
        print(f"Warning: {missing_completion} rows had empty completion text.")


if __name__ == "__main__":
    main()
