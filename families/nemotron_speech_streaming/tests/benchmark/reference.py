#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned NeMo streaming-ASR reference for Accuracy qualification."""

from __future__ import annotations

import json
from typing import Any, Sequence

from tools.benchmark_qualification.references import speech_io


def _text(value: Any) -> str:
    return str(value.text if hasattr(value, "text") else value)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = speech_io.parser().parse_args(argv)
    request = speech_io.load_request(arguments.request)
    samples = speech_io.prepare_samples(request, arguments.output)

    import torch
    from nemo.collections.asr.models import ASRModel

    model_id, revision = speech_io.model_id(request)
    is_nemotron35 = "nemotron-3.5-asr-streaming" in model_id.casefold()
    if is_nemotron35:
        from apps.benchmark.performance.baselines.audio_reference import (
            load_nemotron35_asr_model,
        )

        model = load_nemotron35_asr_model(
            model=model_id,
            revision=revision or "",
            local_files_only=False,
            device="cuda",
        )
    else:
        model = ASRModel.from_pretrained(model_id, map_location="cpu").eval().to("cuda")

    language = request.get("language")
    manifest = arguments.output.parent / "reference-audio" / "manifest.jsonl"
    records = []
    for sample in samples:
        value = {
            "audio_filepath": str(sample.wav_path),
            "duration": float(len(sample.waveform)) / sample.sample_rate,
            "text": "",
        }
        if isinstance(language, str) and language:
            value["lang"] = language
        records.append(value)
    manifest.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records),
        encoding="utf-8",
    )

    original_forward = model.forward
    if is_nemotron35:

        def forward_with_extended_prompt(*args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt")
            if prompt is not None and prompt.shape[1] > 0:
                kwargs = dict(kwargs)
                kwargs["prompt"] = torch.cat((prompt, prompt[:, -1:, :]), dim=1)
            return original_forward(*args, **kwargs)

        model.forward = forward_with_extended_prompt
    try:
        options = {"batch_size": 1}
        if is_nemotron35:
            options["verbose"] = False
        values = model.transcribe(str(manifest), **options)
    finally:
        model.forward = original_forward
    if isinstance(values, tuple):
        values = values[0]
    speech_io.write_result(arguments.output, samples, [_text(value) for value in values])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
