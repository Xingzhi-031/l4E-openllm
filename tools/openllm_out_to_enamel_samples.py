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
    matches = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
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


def _extract_function_by_indent(code: str, target_name: str) -> str:
    """Slice the target function block by indentation, ignoring trailing prose."""
    needle = f"def {target_name}("
    idx = code.rfind(needle)
    if idx < 0:
        return ""
    lines = code[idx:].splitlines()
    if not lines:
        return ""
    kept = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            kept.append(line)
            continue
        if not line.startswith((" ", "\t")):
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _extract_def_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", line):
            start = i
            i += 1
            while i < len(lines):
                cur = lines[i]
                # Next top-level function starts a new block.
                if re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", cur):
                    break
                # Hard stops for common non-code tails.
                if cur.strip().startswith(("```", "# Examples", "# Test", "if __name__")):
                    break
                i += 1
            blocks.append("\n".join(lines[start:i]).strip())
        else:
            i += 1
    return [b for b in blocks if b]


def _choose_best_function_from_blocks(blocks: list[str], entry_point: str) -> tuple[str, bool]:
    funcs: list[ast.FunctionDef] = []
    for block in blocks:
        try:
            tree = ast.parse(block)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                funcs.append(node)
    if not funcs:
        return "", False

    target_funcs = [fn for fn in funcs if fn.name == entry_point]
    if target_funcs:
        best_target = max(target_funcs, key=_function_score)
        if _function_score(best_target) > 0 and not _is_doc_or_pass_only(best_target):
            return ast.unparse(best_target).strip(), False

    nontrivial = [fn for fn in funcs if _function_score(fn) > 0 and not _is_doc_or_pass_only(fn)]
    if not nontrivial:
        # Last resort: keep best target even if weak.
        if target_funcs:
            best_target = max(target_funcs, key=_function_score)
            return ast.unparse(best_target).strip(), False
        best_any = max(funcs, key=_function_score)
        return ast.unparse(best_any).strip(), best_any.name != entry_point

    best_fn = max(nontrivial, key=_function_score)
    best_src = ast.unparse(best_fn).strip()
    if best_fn.name == entry_point:
        return best_src, False
    wrapper = (
        f"\n\ndef {entry_point}(*args, **kwargs):\n"
        f"    return {best_fn.name}(*args, **kwargs)\n"
    )
    return (best_src + wrapper).strip(), True


def _trim_target_function_block(code: str, entry_point: str) -> str:
    pattern = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", flags=re.M)
    match = pattern.search(code)
    if not match:
        return ""
    lines = code[match.start() :].splitlines()
    kept: list[str] = []
    for i, ln in enumerate(lines):
        if i > 0 and re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", ln):
            break
        if ln.strip().startswith(("```", "# Examples", "# Test", "if __name__")):
            break
        kept.append(ln)
    return "\n".join(kept).strip()


def _normalize_body_to_prompt(prompt: str, body_text: str) -> str:
    body = textwrap.dedent(body_text).strip("\n")
    if not body:
        return prompt
    indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
    return prompt if not indented else (prompt + "\n" + indented)


def normalize_solution(prompt: str, entry_point: str, completion: str) -> str:
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    prompt_clean = prompt.rstrip("\n")
    code = extract_code_block(completion)
    prompt = prompt_clean
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
        # If whole text is unparsable, salvage individual def blocks.
        best_from_blocks, wrapped_from_blocks = _choose_best_function_from_blocks(
            _extract_def_blocks(code), entry_point
        )
        if best_from_blocks:
            normalized = best_from_blocks
            wrapped = wrapped or wrapped_from_blocks

    if not normalized:
        salvaged = _extract_function_by_indent(code, entry_point)
        if salvaged:
            try:
                tree2 = ast.parse(salvaged)
                tgt = [n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == entry_point
                       and _function_score(n) > 0 and not _is_doc_or_pass_only(n)]
                if tgt:
                    normalized = ast.unparse(tgt[0]).strip()
            except Exception:
                pass

    if not normalized:
        if target_def in code:
            normalized = _trim_target_function_block(code, entry_point)
        elif re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\(", code, flags=re.M):
            normalized = code.strip()
        else:
            normalized = _normalize_body_to_prompt(prompt, code)

    if wrapped:
        # keep marker for debug (no side effect on execution)
        normalized = normalized + "\n"

    if not normalized:
        normalized = _normalize_body_to_prompt(prompt, code)

    full = IMPORT_PKG + "\n" + normalized.strip() + "\n"
    try:
        ast.parse(full)
    except SyntaxError:
        # Last salvage: try extracting a cleaner function block before giving up.
        trimmed = _trim_target_function_block(code, entry_point)
        if trimmed:
            retry = IMPORT_PKG + "\n" + trimmed.strip() + "\n"
            try:
                ast.parse(retry)
                full = retry
            except SyntaxError:
                body_retry = _normalize_body_to_prompt(prompt, code)
                retry2 = IMPORT_PKG + "\n" + body_retry.strip() + "\n"
                try:
                    ast.parse(retry2)
                    full = retry2
                except SyntaxError:
                    full = IMPORT_PKG + "\n" + prompt.strip() + "\n"
        else:
            body_retry = _normalize_body_to_prompt(prompt, code)
            retry2 = IMPORT_PKG + "\n" + body_retry.strip() + "\n"
            try:
                ast.parse(retry2)
                full = retry2
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
