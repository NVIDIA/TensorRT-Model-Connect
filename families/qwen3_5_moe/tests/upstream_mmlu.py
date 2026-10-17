# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full pinned MMLU batch-one companion evidence, not a replacement for HF quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from families.qwen3_5_moe.edge_llm.builder import EDGE_REVISION

REQUEST_COUNT = 14042
THRESHOLD = 0.35
CONTROLS = {
    "batch_size": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_generate_length": 1,
    "apply_chat_template": False,
}
ERROR_TEXT = "TensorRT Edge LLM cannot handle this request. Fails."


def raw_prompt(row: dict) -> str:
    """Return exact raw bytes for legacy split prompts or the0.11 single-user layout."""
    if set(row) != {"messages", "answer", "subject"}:
        raise ValueError("Unexpected MMLU row fields")
    messages = row["messages"]
    if not isinstance(messages, list) or [m.get("role") for m in messages] not in (
        ["system", "user"], ["user"]
    ):
        raise ValueError("Expected ordered system/user messages or one provider raw user message")
    if any(set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages):
        raise ValueError("MMLU requires original plain-text messages")
    if row["answer"] not in ("A", "B", "C", "D") or not isinstance(row["subject"], str):
        raise ValueError("Invalid MMLU answer/subject")
    return "".join(message["content"] for message in messages)


def validate_dataset(dataset: dict) -> list[dict]:
    """Require the entire frozen workload and controls, returning its ordered rows."""
    if set(dataset) != {*CONTROLS, "requests"}:
        raise ValueError("Unexpected dataset controls")
    for key, value in CONTROLS.items():
        if dataset[key] != value or type(dataset[key]) is not type(value):
            raise ValueError(f"Changed upstream MMLU control: {key}")
    rows = dataset["requests"]
    if not isinstance(rows, list) or len(rows) != REQUEST_COUNT:
        raise ValueError(f"Require all {REQUEST_COUNT} MMLU requests")
    for row in rows:
        raw_prompt(row)
    return rows


def public_requests(dataset: dict) -> list[dict]:
    """Convert every original row to the persistent public driver without whitespace changes."""
    config = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "use_chat_template": False,
        "enable_thinking": False,
    }
    return [
        {"id": str(index), "prompt": raw_prompt(row), "config": dict(config)}
        for index, row in enumerate(validate_dataset(dataset))
    ]


def checked_predictions(dataset: dict, report: dict, kind: str) -> dict:
    """Validate complete ordered public/native results before exposing them to upstream scoring.

    Args:
        dataset: Complete unchanged upstream input document.
        report: Actual persistent driver or native llm_inference JSON.
        kind: Either public or edge, identifying the corresponding output schema.

    Returns:
        Official scorer input with every response retained.

    Raises:
        ValueError: Partial, reordered, failed or mismatched execution evidence.
    """
    rows = validate_dataset(dataset)
    if kind not in ("public", "edge"):
        raise ValueError("Unknown result kind")
    if kind == "public" and (
        report.get("passed") is not True
        or report.get("backend") != "edge_llm"
        or report.get("edge_revision") != EDGE_REVISION
        or report.get("task") != "text_generation"
    ):
        raise ValueError("Require successful public Edge text-task execution")
    results = report.get("results" if kind == "public" else "responses")
    if not isinstance(results, list) or len(results) != len(rows):
        raise ValueError("Missing or extra MMLU responses")
    checked = []
    for index, (row, result) in enumerate(zip(rows, results, strict=True)):
        text = result.get("text" if kind == "public" else "output_text")
        if "error" in result or not isinstance(text, str) or text == ERROR_TEXT:
            raise ValueError(f"Failed MMLU response {index}")
        if kind == "public":
            ids = result.get("token_ids")
            if (
                result.get("id") != str(index)
                or not isinstance(ids, list)
                or len(ids) != 1
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise ValueError(f"Public order/output-budget mismatch at {index}")
        else:
            messages = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in row["messages"]
            ]
            if (
                type(result.get("request_idx")) is not int
                or result["request_idx"] != index
                or result.get("batch_idx") != 0
                or result.get("messages") != messages
                or result.get("formatted_complete_request") != raw_prompt(row)
                or result.get("finish_reason") not in ("max-length", "end-of-sequence")
            ):
                raise ValueError(f"Native input/order/completion mismatch at {index}")
        checked.append({"request_idx": index, "messages": row["messages"], "output_text": text})
    return {"responses": checked}


def score_report(stdout: str) -> dict:
    """Read the unmodified scorer's exact numerator/denominator; never accept filtered subsets."""
    matches = re.findall(r"^Overall Accuracy:.* - (\d+)/(\d+) correct$", stdout, re.MULTILINE)
    if len(matches) != 1 or "Skipped " in stdout:
        raise ValueError("Missing/ambiguous or filtered upstream score")
    correct, total = map(int, matches[0])
    if total != REQUEST_COUNT or not 0 <= correct <= total:
        raise ValueError("Scorer denominator is not the complete workload")
    accuracy = correct / total
    return {
        "passed": accuracy >= THRESHOLD,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "threshold": THRESHOLD,
        "batch_size": 1,
        "listed_batch8_qualification": False,
        "baseline_regression": "separate gate, not implied",
        "independent_hf_quality": "separate gate, not implied",
    }


def main() -> None:
    """Prepare full public inputs or validate/score completed output using pinned Edge source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--kind", choices=("public", "edge"))
    parser.add_argument("--edge-source", type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    if args.action == "prepare":
        value = public_requests(dataset)
    else:
        if args.results is None or args.kind is None or args.edge_source is None:
            parser.error("score requires --results, --kind and --edge-source")
        report = json.loads(args.results.read_text())
        predictions = checked_predictions(dataset, report, args.kind)
        inputs = args.output.with_suffix(".scorer-input.json")
        inputs.write_text(json.dumps(predictions) + "\n")
        scorer = args.edge_source / "examples/accuracy/scripts/calculate_correctness.py"
        revision = subprocess.check_output(
            ["git", "-C", str(args.edge_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != EDGE_REVISION:
            raise ValueError("Reference scorer source is not the pinned Edge revision")
        subprocess.run(
            [
                "git",
                "-C",
                str(args.edge_source),
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                "examples/accuracy/scripts/calculate_correctness.py",
            ],
            check=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--predictions_file",
                str(inputs),
                "--answers_file",
                str(args.dataset),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        args.output.with_suffix(".scorer.log").write_text(completed.stdout)
        completed.check_returncode()
        value = score_report(completed.stdout)
        value.update(
            {"kind": args.kind, "dataset": str(args.dataset), "results": str(args.results)}
        )
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    if args.action == "score" and not value["passed"]:
        raise ValueError("Full MMLU correctness failed the unchanged upstream threshold")


if __name__ == "__main__":
    main()
