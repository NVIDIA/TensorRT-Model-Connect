# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate conversation goldens with the pinned original tokenizer template."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    history = [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "<think>old reasoning</think>\u2003 First answer.\u00a0"},
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "<think>latest reasoning</think> Latest answer."},
        {"role": "user", "content": "Final question."},
    ]
    typed = [dict(message) for message in history]
    typed[1] = {
        "role": "assistant",
        "reasoning": "old reasoning",
        "content": "\u2003 First answer.\u00a0",
    }
    typed[3] = {"role": "assistant", "reasoning": "latest reasoning", "content": " Latest answer."}
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    empty_system = [{"role": "system", "content": ""}, {"role": "user", "content": "Hello."}]
    cases = [
        {"name": "discard_old_raw_reasoning", "messages": history},
        {"name": "keep_old_raw_reasoning", "messages": history, "keep_past_thinking": True},
        {"name": "typed_reasoning", "messages": typed},
        {"name": "empty_system_with_tools", "messages": empty_system, "tools": tools},
        {"name": "empty_system", "messages": empty_system},
    ]
    for case in cases:
        messages = []
        for message in case["messages"]:
            content = message["content"]
            if "reasoning" in message:
                content = "<think>" + message["reasoning"] + "</think>" + content
            messages.append({"role": message["role"], "content": content})
        case["expected"] = tokenizer.apply_chat_template(
            messages,
            tools=case.get("tools"),
            keep_past_thinking=case.get("keep_past_thinking", False),
            add_generation_prompt=True,
            tokenize=False,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "checkpoint": "LiquidAI/LFM2.5-VL-1.6B",
                "revision": "919fde3d022e3f90a4716006f993938ee8c2eb97",
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
