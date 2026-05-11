import argparse
import ast
import contextlib
import io
import json
import os
import re
import statistics


FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
EXPLAIN_RE = re.compile(
    r"(This solution|Explanation|Here is|In summary|复杂度|time complexity|space complexity)",
    re.I,
)


def extract_code(text: str) -> str:
    m = FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def eval_label(rate: float, good: float, warn: float) -> str:
    if rate >= good:
        return "good"
    if rate >= warn:
        return "medium"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality check for L4E open-LLM output JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to generated JSON result file",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size for quick test pass rate")
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(base_dir, "cache", "open_llm", "sample_Qwen2.5-0.5B-Instruct.json")

    data = json.load(open(input_path, "r", encoding="utf-8"))
    total = len(data)

    lengths = [len((x.get("completion") or "")) for x in data]
    nonempty_items = [x for x in data if (x.get("completion") or "").strip()]
    nonempty = len(nonempty_items)
    empty = total - nonempty

    texts = [(x.get("completion") or "") for x in nonempty_items]
    explain_count = sum(1 for s in texts if EXPLAIN_RE.search(s))
    fence_count = sum(1 for s in texts if "```" in s)

    syntax_ok = 0
    for x in nonempty_items:
        code = extract_code(x.get("completion") or "")
        try:
            ast.parse(code)
            syntax_ok += 1
        except Exception:
            pass
    syntax_fail = nonempty - syntax_ok
    syntax_rate = (syntax_ok / nonempty) if nonempty else 0.0

    # 抽样功能通过率（屏蔽被测代码中的 print 噪音）
    sample_n = min(args.sample_size, total)
    sample_pass = 0
    sample_tested = 0
    sample_skipped_no_cases = 0
    for x in data[:sample_n]:
        s = (x.get("completion") or "").strip()
        tests = (x.get("small_test_cases") or "").strip()
        if not s:
            continue
        if not tests:
            sample_skipped_no_cases += 1
            continue
        code = extract_code(s)
        ns = {}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(code, ns, ns)
                exec(tests, ns, ns)
            sample_pass += 1
        except Exception:
            pass
        sample_tested += 1
    sample_pass_rate = (sample_pass / sample_tested) if sample_tested else 0.0

    print("========== L4E Quality Report ==========")
    print(f"file: {input_path}")
    print()
    print("[1] Coverage and Length")
    print(f"- total: {total}")
    print(f"- nonempty: {nonempty}")
    print(f"- empty: {empty} ({(empty/total if total else 0):.2%})")
    print(f"- avg_len: {round(statistics.mean(lengths), 1) if lengths else 0}")
    print(f"- median_len: {statistics.median(lengths) if lengths else 0}")
    print(f"- p90_len: {sorted(lengths)[int(total * 0.9) - 1] if total else 0}")
    print()
    print("[2] Text Noise")
    print(f"- with_explain_markers: {explain_count} ({(explain_count/nonempty if nonempty else 0):.2%})")
    print(f"- with_code_fence: {fence_count} ({(fence_count/nonempty if nonempty else 0):.2%})")
    print()
    print("[3] Syntax Quality")
    print(f"- syntax_ok: {syntax_ok}")
    print(f"- syntax_fail: {syntax_fail}")
    print(f"- syntax_ok_rate: {syntax_rate:.2%} ({eval_label(syntax_rate, 0.85, 0.60)})")
    print()
    print("[4] Sample Functional Quality")
    print(f"- sample_tested: {sample_tested} (first {sample_n} items)")
    print(f"- sample_skipped_no_small_test_cases: {sample_skipped_no_cases}")
    print(f"- sample_pass: {sample_pass}")
    print(f"- sample_pass_rate: {sample_pass_rate:.2%} ({eval_label(sample_pass_rate, 0.50, 0.25)})")
    print()
    print("Notes:")
    print("- syntax_ok_rate: share of outputs that can be parsed as Python code.")
    print("- sample_pass_rate: approximate real usability on bundled small tests.")
    print("- If sample_pass_rate is low, run extraction/cleanup first, then iterative fixing.")


if __name__ == "__main__":
    main()
