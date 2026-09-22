#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned NeMo streaming-ASR reference for Accuracy qualification."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import atexit
import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.references import speech_io
from qualification_tests.benchmark_qualification.performance import reference_harness
from qualification_tests.benchmark_qualification.performance.references.audio_reference import (
    load_audio_mono,
    resample_linear,
    transcription_text,
    write_wav_pcm16,
)


def _text(value: Any) -> str:
    return str(value.text if hasattr(value, "text") else value)


def _accuracy(argv: Sequence[str] | None = None) -> int:
    arguments = speech_io.parser().parse_args(argv)
    request = speech_io.load_request(arguments.request)
    samples = speech_io.prepare_samples(request, arguments.output)

    import torch
    from nemo.collections.asr.models import ASRModel

    model_id, revision = speech_io.model_id(request)
    is_nemotron35 = "nemotron-3.5-asr-streaming" in model_id.casefold()
    if is_nemotron35:
        from qualification_tests.benchmark_qualification.performance.references.audio_reference import (
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


def _disable_cuda_graphs(model: Any) -> None:
    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding is not None and hasattr(decoding, "use_cuda_graph_decoder"):
        decoding.use_cuda_graph_decoder = False
    change_strategy = getattr(model, "change_decoding_strategy", None)
    if callable(change_strategy) and decoding is not None:
        change_strategy(decoding_cfg=decoding)


def _performance_session(
    arguments: Any,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "pytorch-eager" or arguments.precision != "fp32":
        raise ValueError("Nemotron streaming ASR reference requires pytorch-eager fp32")
    import torch
    from nemo.collections.asr.models import ASRModel

    is_nemotron35 = "nemotron-3.5-asr-streaming" in arguments.model.casefold()
    if is_nemotron35:
        from qualification_tests.benchmark_qualification.performance.references.audio_reference import (
            load_nemotron35_asr_model,
        )

        model = load_nemotron35_asr_model(
            model=arguments.model,
            revision=arguments.revision or "",
            local_files_only=arguments.local_files_only,
            device="cuda",
        )
    else:
        model = ASRModel.from_pretrained(arguments.model, map_location="cpu").eval().to("cuda")
    _disable_cuda_graphs(model)
    audio, source_rate = load_audio_mono(str(request["audio_path"]))
    target_rate = 16_000
    audio = resample_linear(audio, source_rate, target_rate)
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    atexit.register(Path(temporary.name).unlink, missing_ok=True)
    write_wav_pcm16(Path(temporary.name), audio, target_rate)
    manifest = Path(temporary.name).with_suffix(".jsonl")
    atexit.register(manifest.unlink, missing_ok=True)
    record: dict[str, Any] = {
        "audio_filepath": temporary.name,
        "duration": float(len(audio)) / target_rate,
        "text": "",
    }
    language = str(request.get("language", "") or "")
    if language and language != "auto":
        record["lang"] = language
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    if is_nemotron35:
        original_forward = model.forward

        def forward_with_extended_prompt(*args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt")
            if prompt is not None and prompt.shape[1] > 0:
                kwargs = dict(kwargs)
                kwargs["prompt"] = torch.cat((prompt, prompt[:, -1:, :]), dim=1)
            return original_forward(*args, **kwargs)

        model.forward = forward_with_extended_prompt

    def invoke() -> Mapping[str, Any]:
        options: dict[str, Any] = {"batch_size": 1}
        if is_nemotron35:
            options["verbose"] = False
        values = model.transcribe(str(manifest), **options)
        return {"text": transcription_text(values), "output_tokens": None}

    return reference_harness.Session(
        invoke,
        "nemo",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--request" in values:
        return _accuracy(values)
    return reference_harness.run(values, description=__doc__, load=_performance_session)


if __name__ == "__main__":
    raise SystemExit(main())
