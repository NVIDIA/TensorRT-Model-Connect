# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture pinned HF template outputs without loading weights or using a GPU."""

import argparse
import json
from pathlib import Path


def cases():
    tool = {"name": "set_temperature", "description": "Set cabin temperature.",
            "parameters": {"type": "object", "properties": {"temperature": {"type": "number"}},
                           "required": ["temperature"]}}
    call = {"function": {"name": "set_temperature", "arguments": {"temperature": 22}}}
    user = {"role": "user", "content": "Set the cabin temperature to 22 degrees Celsius."}
    yield {"name": "single-user", "messages": [user]}
    yield {"name": "system-tools", "messages": [{"role": "system", "content": "Use tools."}, user], "tools": [tool]}
    yield {"name": "tools-without-system", "messages": [user], "tools": [tool]}
    yield {"name": "json-tool-request", "messages": [{"role": "system", "content": "Return tool calls as JSON objects with name and arguments."}, user], "tools": [tool]}
    yield {"name": "tool-result", "messages": [user, {"role": "assistant", "tool_calls": [call]},
           {"role": "tool", "content": '{"temperature":22,"success":true}'}], "tools": [tool]}
    prior = [user, {"role": "assistant", "reasoning": "Use the supplied tool.", "content": "Done."},
             {"role": "user", "content": "Now set it to 23."}]
    yield {"name": "strip-prior-reasoning", "messages": prior}
    yield {"name": "preserve-prior-reasoning", "messages": prior, "preserve_thinking": True}
    yield {"name": "strip-embedded-reasoning", "messages": [user,
           {"role": "assistant", "content": "<think>old reasoning</think>  Done.  "},
           {"role": "user", "content": "Now set it to 23."}]}
    yield {"name": "strip-unicode-whitespace", "messages": [user,
           {"role": "assistant", "content": "<think>old reasoning</think>\u00a0\u3000\u001c\u001d\u001e\u001f Done.\u001f\u001e\u001d\u001c\u3000\u00a0"},
           {"role": "user", "content": "Now set it to 23."}]}
    yield {"name": "current-assistant-reasoning", "messages": [user,
           {"role": "assistant", "reasoning": "Use the tool.", "content": "", "tool_calls": [call]},
           {"role": "tool", "content": "Success"}]}
    arguments = {"text": "O'Brien\\route\nsecond\rline", "enabled": True, "empty": None,
                 "list": ["O'Brien", False, None, {"nested": "日本語"}], "object": {"a": [1, 2]}}
    yield {"name": "argument-format", "messages": [user, {"role": "assistant", "content": "Calling.",
           "tool_calls": [{"function": {"name": "record", "arguments": arguments}}]}]}
    yield {"name": "parallel-tools", "messages": [user, {"role": "assistant", "tool_calls": [call, call]},
           {"role": "tool", "content": "First success"}, {"role": "tool", "content": "Second success"}]}
    yield {"name": "unicode-system", "messages": [{"role": "system", "content": "Réponds en français."},
           {"role": "developer", "content": "Use Unicode: العربية हिन्दी ไทย 日本語."}, user]}


if __name__ == "__main__":
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    records = []
    for case in cases():
        options = {key: case[key] for key in ("tools", "preserve_thinking") if key in case}
        case["rendered"] = tokenizer.apply_chat_template(case["messages"], tokenize=False,
                                                          add_generation_prompt=True, **options)
        case["token_ids"] = tokenizer(case["rendered"], add_special_tokens=False).input_ids
        records.append(case)
    output = Path(__file__).with_name("fixtures") / "lfm25-conversation.json"
    output.write_text(json.dumps({
        "revision": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
        "template": (args.checkpoint / "chat_template.jinja").read_bytes().decode("utf-8"),
        "cases": records,
    }, indent=2, ensure_ascii=False) + "\n")
