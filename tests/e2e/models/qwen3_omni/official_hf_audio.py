#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Transformers Qwen3-Omni text-to-audio path."""

from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path
from typing import Sequence


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


def _write_wav(path: Path, audio, sample_rate: int) -> int:
    import numpy as np

    samples = audio.detach().cpu().float().numpy()
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("Qwen3-Omni produced empty or non-finite audio")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return int(samples.size)


def run(arguments: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
        __version__ as transformers_version,
    )

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    common = {
        "revision": arguments.revision or None,
        "local_files_only": arguments.local_files_only,
    }
    common = {key: value for key, value in common.items() if value is not None}
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        arguments.model,
        **common,
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model,
        **common,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        enable_audio_output=True,
    ).eval()
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": arguments.prompt}],
        },
    ]
    processor_template = getattr(processor, "chat_template", None)
    tokenizer_template = getattr(
        getattr(processor, "tokenizer", None),
        "chat_template",
        None,
    )
    template = (
        processor_template
        if isinstance(processor_template, str) and processor_template.strip()
        else tokenizer_template
        if isinstance(tokenizer_template, str) and tokenizer_template.strip()
        else TEXT_CHAT_TEMPLATE
    )
    inputs = processor.apply_chat_template(
        conversation,
        chat_template=template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    with torch.inference_mode():
        text_ids, audio = model.generate(
            **inputs,
            thinker_max_new_tokens=arguments.thinker_max_new_tokens,
            talker_max_new_tokens=arguments.talker_max_new_tokens,
            thinker_do_sample=False,
            talker_do_sample=False,
            speaker=arguments.speaker,
        )
    decoded_text = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
    )[0]
    sample_rate = 24_000
    num_samples = _write_wav(arguments.audio_output, audio, sample_rate)
    resolved_revision = str(
        arguments.revision
        or getattr(model.config, "_commit_hash", "")
        or "unresolved"
    )
    arguments.metadata_output.write_text(
        json.dumps(
            {
                "model_id": arguments.model,
                "resolved_revision": resolved_revision,
                "transformers_version": transformers_version,
                "system_prompt": SYSTEM_PROMPT,
                "prompt": arguments.prompt,
                "speaker": arguments.speaker,
                "seed": arguments.seed,
                "thinker_max_new_tokens": arguments.thinker_max_new_tokens,
                "talker_max_new_tokens": arguments.talker_max_new_tokens,
                "thinker_do_sample": False,
                "talker_do_sample": False,
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "decoded_text": decoded_text,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=16)
    parser.add_argument("--talker-max-new-tokens", type=int, default=32)
    parser.add_argument("--audio-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
