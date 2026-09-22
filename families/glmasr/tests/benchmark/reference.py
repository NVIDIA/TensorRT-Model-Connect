#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official GLM-ASR reference for Accuracy qualification."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.references import speech_io
from qualification_tests.benchmark_qualification.performance import reference_harness
from qualification_tests.benchmark_qualification.performance.references.audio_reference import (
    load_audio_mono,
    resample_linear,
)


def _accuracy(argv: Sequence[str] | None = None) -> int:
    arguments = speech_io.parser().parse_args(argv)
    request = speech_io.load_request(arguments.request)
    samples = speech_io.prepare_samples(request, arguments.output)

    import torch
    from transformers import GlmAsrForConditionalGeneration, GlmAsrProcessor

    model_id, revision = speech_io.model_id(request)
    options = {"revision": revision} if revision else {}
    processor = GlmAsrProcessor.from_pretrained(model_id, **options)
    model = (
        GlmAsrForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=speech_io.torch_dtype(torch, request),
            **options,
        )
        .eval()
        .to("cuda")
    )
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


def _performance_session(
    arguments: Any,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "hf-eager":
        raise ValueError("GLM-ASR reference requires hf-eager mode")
    import torch
    from transformers import GlmAsrForConditionalGeneration, GlmAsrProcessor

    options = {"revision": arguments.revision} if arguments.revision else {}
    processor = GlmAsrProcessor.from_pretrained(arguments.model, **options)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        arguments.precision
    ]
    model = (
        GlmAsrForConditionalGeneration.from_pretrained(
            arguments.model, torch_dtype=dtype, **options
        )
        .eval()
        .to("cuda")
    )
    audio, source_rate = load_audio_mono(str(request["audio_path"]))
    target_rate = 16_000
    audio = resample_linear(audio, source_rate, target_rate)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
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
        sampling_rate=target_rate,
    ).to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    max_new_tokens = int(request.get("max_new_tokens", 128))

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        token_ids = [int(token) for token in generated[0, prompt_tokens:].cpu().tolist()]
        return {
            "text": processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            "token_ids": token_ids,
            "output_tokens": len(token_ids),
        }

    return reference_harness.Session(invoke, "transformers")


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--request" in values:
        return _accuracy(values)
    return reference_harness.run(values, description=__doc__, load=_performance_session)


if __name__ == "__main__":
    raise SystemExit(main())
