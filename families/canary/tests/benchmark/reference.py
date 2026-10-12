#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned NeMo Canary reference for Accuracy qualification."""

from __future__ import annotations

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

    from huggingface_hub import snapshot_download
    from nemo.collections.asr.models import ASRModel

    model_id, revision = speech_io.model_id(request)
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.nemo"],
    )
    archives = sorted(Path(snapshot).glob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"Canary NeMo archive is missing for {model_id}")
    model = ASRModel.restore_from(str(archives[0]), map_location="cpu").eval().to("cuda")
    values = model.transcribe([str(sample.wav_path) for sample in samples], batch_size=1)
    if isinstance(values, tuple):
        values = values[0]
    speech_io.write_result(arguments.output, samples, [_text(value) for value in values])
    return 0


def _performance_session(
    arguments: Any,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "pytorch-eager" or arguments.precision != "fp32":
        raise ValueError("Canary reference requires pytorch-eager fp32")
    from huggingface_hub import snapshot_download
    from nemo.collections.asr.models import ASRModel

    snapshot = snapshot_download(
        repo_id=arguments.model,
        revision=arguments.revision,
        allow_patterns=["*.nemo"],
        local_files_only=arguments.local_files_only,
    )
    archives = sorted(Path(snapshot).glob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"Canary NeMo archive is missing for {arguments.model}")
    model = ASRModel.restore_from(str(archives[0]), map_location="cpu").cpu().eval()
    audio, source_rate = load_audio_mono(str(request["audio_path"]))
    target_rate = 16_000
    audio = resample_linear(audio, source_rate, target_rate)
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    atexit.register(Path(temporary.name).unlink, missing_ok=True)
    write_wav_pcm16(Path(temporary.name), audio, target_rate)

    def invoke() -> Mapping[str, Any]:
        values = model.transcribe([temporary.name], batch_size=1)
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
