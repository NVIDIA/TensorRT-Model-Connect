#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official GLM-ASR reference for Accuracy qualification."""

from __future__ import annotations

from typing import Sequence

from tools.benchmark_qualification.references import speech_io


def main(argv: Sequence[str] | None = None) -> int:
    arguments = speech_io.parser().parse_args(argv)
    request = speech_io.load_request(arguments.request)
    samples = speech_io.prepare_samples(request, arguments.output)

    import torch
    from transformers import GlmAsrForConditionalGeneration, GlmAsrProcessor

    model_id, revision = speech_io.model_id(request)
    options = {"revision": revision} if revision else {}
    processor = GlmAsrProcessor.from_pretrained(model_id, **options)
    model = GlmAsrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=speech_io.torch_dtype(torch, request),
        **options,
    ).eval().to("cuda")
    max_new_tokens = int(request.get("max_new_tokens", 128))
    texts = []
    for sample in samples:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": sample.waveform},
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            sampling_rate=sample.sample_rate,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        texts.append(processor.tokenizer.decode(new_tokens, skip_special_tokens=True))
    speech_io.write_result(arguments.output, samples, texts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
