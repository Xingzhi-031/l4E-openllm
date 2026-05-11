import argparse
import json
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(sample: Dict[str, Any]) -> str:
    if "markdown_description" in sample:
        tests = sample.get("small_test_cases", "")
        return (
            "Please complete Python code based on the task description and test cases.\n"
            f"# Task description:\n{sample['markdown_description']}\n"
            f"{tests}\n"
            "# Solution:\n"
        )
    if "prompt" in sample:
        tests = sample.get("test_list", [])
        tests_text = "\n".join(tests) if isinstance(tests, list) else str(tests)
        return (
            "Please complete Python code based on the task description and test cases.\n"
            f"# Task description:\n{sample['prompt']}\n"
            f"{tests_text}\n"
            "# Solution:\n"
        )
    if "description" in sample:
        return (
            "Please complete Python code based on the task description.\n"
            f"# Task description:\n{sample['description']}\n"
            "# Solution:\n"
        )
    return "Please complete Python code.\n# Solution:\n"


def batched(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def batched_indices(indices: List[int], batch_size: int):
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def generate_batch(
    batch: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> List[str]:
    prompts = [build_prompt(x) for x in batch]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = encoded.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned = []
    for prompt, text in zip(prompts, decoded):
        if text.startswith(prompt):
            cleaned.append(text[len(prompt) :].strip())
        else:
            cleaned.append(text.strip())
    return cleaned


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    if args.output_path:
        out_path = args.output_path
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base_dir, "cache", "open_llm")
        os.makedirs(out_dir, exist_ok=True)
        ckpt_name = args.checkpoint.split("/")[-1]
        out_path = os.path.join(out_dir, f"sample_{ckpt_name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    resumed = 0
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            if isinstance(prev_data, list):
                for i in range(min(len(dataset), len(prev_data))):
                    prev_completion = prev_data[i].get("completion", "")
                    if isinstance(prev_completion, str) and prev_completion.strip():
                        dataset[i]["completion"] = prev_completion
                        resumed += 1
        except Exception:
            # Ignore broken/incompatible old output and regenerate from current dataset.
            pass

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    remaining_indices = [
        i
        for i, sample in enumerate(dataset)
        if not isinstance(sample.get("completion", ""), str)
        or not sample.get("completion", "").strip()
    ]
    print(
        f"Resume detected: {resumed}/{len(dataset)} completed, "
        f"remaining: {len(remaining_indices)}"
    )

    generated_since_last_save = 0
    if len(remaining_indices) > 0:
        progress = tqdm(
            batched_indices(remaining_indices, args.batch_size),
            total=(len(remaining_indices) + args.batch_size - 1) // args.batch_size,
            desc="Generating",
        )
        for index_batch in progress:
            batch = [dataset[i] for i in index_batch]
            completions = generate_batch(batch, model, tokenizer, args.max_new_tokens)
            for i, comp in zip(index_batch, completions):
                dataset[i]["completion"] = comp
                generated_since_last_save += 1

            if generated_since_last_save >= args.save_every:
                save_json(out_path, dataset)
                generated_since_last_save = 0

    save_json(out_path, dataset)
    print(f"Saved {len(dataset)} samples to: {out_path}")


if __name__ == "__main__":
    main()
