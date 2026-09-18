#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned NeMo Canary reference for Accuracy qualification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tools.benchmark_qualification.references import speech_io


def _text(value: Any) -> str:
    return str(value.text if hasattr(value, "text") else value)


def main(argv: Sequence[str] | None = None) -> int:
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
    model = ASRModel.restore_from(str(archives[0]), map_location="cpu").cpu().eval()
    values = model.transcribe([str(sample.wav_path) for sample in samples], batch_size=1)
    if isinstance(values, tuple):
        values = values[0]
    speech_io.write_result(arguments.output, samples, [_text(value) for value in values])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
