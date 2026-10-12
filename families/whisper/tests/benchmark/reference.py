#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Transformers Whisper reference for Accuracy qualification."""

from __future__ import annotations

from typing import Sequence

from qualification_tests.benchmark_qualification.references import speech_io


def main(argv: Sequence[str] | None = None) -> int:
    arguments = speech_io.parser().parse_args(argv)
    request = speech_io.load_request(arguments.request)
    samples = speech_io.prepare_samples(request, arguments.output)

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id, revision = speech_io.model_id(request)
    options = {"revision": revision} if revision else {}
    processor = AutoProcessor.from_pretrained(model_id, **options)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=speech_io.torch_dtype(torch, request),
        **options,
    ).eval().to("cuda")
    max_new_tokens = int(request.get("max_new_tokens", 128))
    language = request.get("language")
    texts = []
    for sample in samples:
        inputs = processor(
            sample.waveform,
            sampling_rate=sample.sample_rate,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        inputs = {
            key: (
                value.to(device=model.device, dtype=model_dtype)
                if value.is_floating_point()
                else value.to(device=model.device)
            )
            for key, value in inputs.items()
        }
        generation = {"max_new_tokens": max_new_tokens}
        if isinstance(language, str) and language:
            generation["language"] = language
        with torch.inference_mode():
            token_ids = model.generate(**inputs, **generation)
        texts.append(processor.batch_decode(token_ids, skip_special_tokens=True)[0])
    speech_io.write_result(arguments.output, samples, texts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
