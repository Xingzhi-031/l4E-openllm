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


def _strip_docstring_body(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _function_score(fn: ast.FunctionDef) -> int:
    body = _strip_docstring_body(fn.body[:])
    if not body:
        return 0
    score = 0
    module = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(
            node,
            (
                ast.Return,
                ast.For,
                ast.While,
                ast.If,
                ast.Assign,
                ast.AugAssign,
                ast.Call,
                ast.Try,
                ast.With,
                ast.Raise,
                ast.Assert,
            ),
        ):
            score += 1
    return score


def _is_doc_or_pass_only(fn: ast.FunctionDef) -> bool:
    body = _strip_docstring_body(fn.body[:])
    if not body:
        return True
    return all(isinstance(node, ast.Pass) for node in body)


def normalize_solution(prompt: str, entry_point: str, completion: str) -> str:
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    code = extract_code_block(completion)
    prompt = prompt.rstrip("\n")
    target_def = f"def {entry_point}("

    normalized = ""
    wrapped = False

    try:
        tree = ast.parse(code)
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        target_funcs = [fn for fn in funcs if fn.name == entry_point]

        if target_funcs:
            best_target = max(target_funcs, key=_function_score)
            best_target_score = _function_score(best_target)
            if best_target_score > 0 and not _is_doc_or_pass_only(best_target):
                normalized = ast.unparse(best_target).strip()
            else:
                # If the target function is docstring/pass-only, salvage the
                # strongest non-trivial function from the same completion.
                nontrivial = [
                    fn for fn in funcs if _function_score(fn) > 0 and not _is_doc_or_pass_only(fn)
                ]
                if nontrivial:
                    best_fn = max(nontrivial, key=_function_score)
                    best_name = best_fn.name
                    best_src = ast.unparse(best_fn).strip()
                    if best_name == entry_point:
                        normalized = best_src
                    else:
                        wrapper = (
                            f"\n\ndef {entry_point}(*args, **kwargs):\n"
                            f"    return {best_name}(*args, **kwargs)\n"
                        )
                        normalized = (best_src + wrapper).strip()
                        wrapped = True
                else:
                    normalized = ast.unparse(best_target).strip()
        elif funcs:
            # Keep the strongest function and add wrapper to expected entry point.
            best_fn = max(funcs, key=_function_score)
            best_name = best_fn.name
            best_src = ast.unparse(best_fn).strip()
            wrapper = (
                f"\n\ndef {entry_point}(*args, **kwargs):\n"
                f"    return {best_name}(*args, **kwargs)\n"
            )
            normalized = (best_src + wrapper).strip()
            wrapped = True
    except Exception:
        normalized = ""

    if not normalized:
        if target_def in code:
            normalized = code[code.rfind(target_def) :].strip()
        elif re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\(", code, flags=re.M):
            normalized = code.strip()
        else:
            body = textwrap.dedent(code).strip("\n")
            indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
            normalized = prompt if not indented else (prompt + "\n" + indented)

    if wrapped:
        # keep marker for debug (no side effect on execution)
        normalized = normalized + "\n"

    if not normalized:
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
    wrapped_count = 0
    prompt_fallback_count = 0

    for item in rows:
        pid = int(item["problem_id"])
        prompt = str(item["prompt"])
        entry_point = str(item["entry_point"])
        completion = str(item.get(args.completion_key, ""))
        if not completion:
            missing_completion += 1
        code = normalize_solution(prompt=prompt, entry_point=entry_point, completion=completion)
        if f"def {entry_point}(*args, **kwargs)" in code:
            wrapped_count += 1
        # heuristic: if output starts with prompt signature/docstring, we likely fell back
        if code.startswith(IMPORT_PKG) and prompt.strip() in code:
            prompt_fallback_count += 1
        samples[str(pid)] = [code]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} tasks -> {out}")
    if missing_completion:
        print(f"Warning: {missing_completion} rows had empty completion text.")
    print(f"Wrapped with entry-point adapters: {wrapped_count}")
    print(f"Prompt-fallback heuristic count: {prompt_fallback_count}")


if __name__ == "__main__":
    main()
