# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary's native pipeline for the trtmc-perf-serve reference backend (``transcribe``): the NeMo model
restored from the checkpoint's ``.nemo`` archive at its pinned revision, transcribing the 16 kHz mono file, as
the family's benchmark reference does."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping

import soundfile


SAMPLE_RATE = 16_000


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import torch
        from huggingface_hub import snapshot_download
        from nemo.collections.asr.models import ASRModel

        self.spec, self.torch = spec, torch
        snapshot = Path(snapshot_download(spec.model, revision=spec.revision, allow_patterns=["*.nemo"]))
        archives = sorted(snapshot.glob("*.nemo"))
        if not archives:
            raise FileNotFoundError(f"no .nemo archive in {spec.model}")
        self.model = ASRModel.restore_from(str(archives[0]), map_location="cpu").eval().to(spec.device)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        audio = self.host.load_audio(str(self.host.required(request, "audio_path")), SAMPLE_RATE)
        wav = artifact_base.with_suffix(".input.wav")
        wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        precision = (self.torch.autocast("cuda", dtype=self.spec.dtype) if self.spec.precision != "fp32"
                     else contextlib.nullcontext())

        def run() -> Any:
            with precision:
                return self.model.transcribe([str(wav)], batch_size=1)

        values, model_ms = self.host.timed(run)
        value = values[0] if isinstance(values, tuple) else values
        value = value[0] if isinstance(value, list) and value else value
        text = str(getattr(value, "text", value) if not isinstance(value, Mapping) else value.get("text", ""))
        seconds = len(audio) / SAMPLE_RATE
        return self.host.invocation({"text": text, "input_audio_seconds": seconds}, model_ms,
                                    realtime_factor=seconds / (model_ms / 1000.0))
