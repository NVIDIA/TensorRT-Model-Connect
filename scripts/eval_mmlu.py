#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


@dataclass
class MmluExample:
    question: str
    choices: list[str]
    answer_index: int
    subject: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model on MMLU multiple-choice QA. "
            "Supports transformers reference or trtmc binary inference."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "trtmc"],
        default="transformers",
        help="Inference backend to evaluate.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id/path for transformers backend, or model id/path passed to trtmc binary.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation", "dev"],
        help="MMLU split to evaluate.",
    )
    parser.add_argument(
        "--subject",
        default="all",
        help="MMLU subject config (default: all).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Max generated tokens for each answer.",
    )
    parser.add_argument(
        "--trtmc-binary",
        default="./build/trtmc",
        help="Path to trtmc binary when --backend=trtmc.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.35,
        help="Required minimum accuracy. Script exits non-zero if not met.",
    )
    return parser.parse_args()


def load_mmlu_examples(subject: str, split: str, num_samples: int, seed: int) -> list[MmluExample]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - runtime dep
        raise RuntimeError("datasets package is required for MMLU evaluation.") from exc

    ds = load_dataset("cais/mmlu", subject, split=split)
    if len(ds) == 0:
        raise RuntimeError("MMLU dataset split is empty.")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: min(num_samples, len(indices))]

    out: list[MmluExample] = []
    for idx in indices:
        row = ds[int(idx)]
        out.append(
            MmluExample(
                question=str(row["question"]),
                choices=[str(c) for c in row["choices"]],
                answer_index=int(row["answer"]),
                subject=str(row.get("subject", subject)),
            )
        )
    return out


def build_prompt(example: MmluExample) -> str:
    lines = [
        f"Subject: {example.subject}",
        "Choose the best answer (A, B, C, or D).",
        f"Question: {example.question}",
        f"A. {example.choices[0]}",
        f"B. {example.choices[1]}",
        f"C. {example.choices[2]}",
        f"D. {example.choices[3]}",
        "Answer:",
    ]
    return "\n".join(lines)


def extract_choice_letter(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()

    stripped = text.strip().upper()
    for c in stripped:
        if c in "ABCD":
            return c
    return None


def answer_letter_from_index(index: int) -> str:
    return "ABCD"[index]


def iter_progress(examples: Iterable[MmluExample], total: int) -> Iterable[tuple[int, MmluExample]]:
    for i, ex in enumerate(examples, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"[eval] example {i}/{total}", file=sys.stderr)
        yield i, ex


def evaluate_transformers(model_name: str, examples: list[MmluExample], max_new_tokens: int) -> tuple[float, int, int]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dep
        raise RuntimeError("transformers + torch are required for transformers backend.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    answered = 0
    total = len(examples)
    for _, ex in iter_progress(examples, total):
        prompt = build_prompt(ex)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_choice_letter(completion)
        if pred is None:
            continue
        answered += 1
        if pred == answer_letter_from_index(ex.answer_index):
            correct += 1

    denom = max(answered, 1)
    return correct / float(denom), correct, answered


def evaluate_trtmc(
    model_id: str, binary_path: str, examples: list[MmluExample],
    max_new_tokens: int,
) -> tuple[float, int, int]:
    cmd_prefix = [binary_path, "run", model_id, "--prompt", ""]
    cmd_prefix.extend(["--max-new-tokens", str(max_new_tokens)])
    correct = 0
    answered = 0
    total = len(examples)
    for _, ex in iter_progress(examples, total):
        prompt = build_prompt(ex)
        cmd = list(cmd_prefix)
        # Replace the placeholder prompt
        prompt_idx = cmd.index("--prompt") + 1
        cmd[prompt_idx] = prompt
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            continue

        # trtmc run prints generated text to stdout (after [trtmc] status on stderr)
        output_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not output_lines:
            continue
        generated = output_lines[-1]
        pred = extract_choice_letter(generated)
        if pred is None:
            continue
        answered += 1
        if pred == answer_letter_from_index(ex.answer_index):
            correct += 1

    denom = max(answered, 1)
    return correct / float(denom), correct, answered


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise RuntimeError("--num-samples must be > 0")

    examples = load_mmlu_examples(args.subject, args.split, args.num_samples, args.seed)
    if not examples:
        raise RuntimeError("No examples sampled from MMLU.")

    if args.backend == "transformers":
        accuracy, correct, answered = evaluate_transformers(args.model, examples, args.max_new_tokens)
    else:
        accuracy, correct, answered = evaluate_trtmc(
            args.model,
            args.trtmc_binary,
            examples,
            args.max_new_tokens,
        )

    total = len(examples)
    print(f"backend={args.backend}")
    print(f"model={args.model}")
    print(f"subject={args.subject}")
    print(f"split={args.split}")
    print(f"sampled={total}")
    print(f"answered={answered}")
    print(f"correct={correct}")
    print(f"accuracy={accuracy:.4f}")
    print(f"min_accuracy={args.min_accuracy:.4f}")

    if accuracy < args.min_accuracy:
        print("status=FAIL", file=sys.stderr)
        return 1
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
