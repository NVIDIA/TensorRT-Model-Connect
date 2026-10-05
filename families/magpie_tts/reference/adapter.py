# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie TTS's native pipeline for the trtmc-perf-serve reference backend (``generate_audio``): NeMo's
MagpieTTSModel from the checkpoint's ``.nemo`` archive at fp32, English with classifier-free guidance, the
request's seed and decoder-step budget, as the family's benchmark reference runs it. The speaker encoder the
archive names by URL is read from its pinned Hugging Face revision instead."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping


ARCHIVE = "magpie_tts_multilingual_357m.nemo"
SAMPLE_RATE = 22_050
SPEAKER_REPOSITORY = "Edresson/Speaker_Encoder_H_ASP"
SPEAKER_FILENAME = "pytorch_model.bin"
SPEAKER_REVISION = "e9124b5364a2c3e9b4f78da429a33cbca8f8c22b"
SPEAKER_URL = f"https://huggingface.co/{SPEAKER_REPOSITORY}/resolve/main/{SPEAKER_FILENAME}"


def _seed(torch: Any, value: int) -> None:
    import numpy as np

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        if spec.precision != "fp32":
            raise self.host.Error("the Magpie TTS reference runs at fp32")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import fsspec
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.tts.models import MagpieTTSModel

        torch.use_deterministic_algorithms(True)
        speaker = hf_hub_download(SPEAKER_REPOSITORY, SPEAKER_FILENAME, revision=SPEAKER_REVISION)
        original_open = fsspec.open

        def pinned_open(path: Any, *args: Any, **kwargs: Any) -> Any:
            return original_open(speaker if str(path).split("?", 1)[0] == SPEAKER_URL else path, *args, **kwargs)

        fsspec.open = pinned_open
        archive = hf_hub_download(spec.model, ARCHIVE, revision=spec.revision)
        self.torch = torch
        self.model = MagpieTTSModel.restore_from(restore_path=archive).eval().to(spec.device)
        self.default_steps = self.model.inference_parameters.max_decoder_steps

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        steps = int(request.get("max_new_tokens") or 0)
        self.model.inference_parameters.max_decoder_steps = steps if steps > 0 else self.default_steps
        prompt = str(self.host.required(request, "prompt"))

        def run() -> Any:
            _seed(self.torch, int(request.get("seed", 42)))
            return self.model.do_tts(transcript=prompt, language="en", use_cfg=True)

        (audio, length), model_ms = self.host.timed(run)
        count = int(length.item()) if length.numel() else int(audio.numel())
        samples = audio.detach().float().cpu().reshape(-1)[:count]
        return self.host.invocation({**self.host.tensor_observation(samples, artifact_base), "sample_rate": SAMPLE_RATE,
                                     "audio_seconds": count / SAMPLE_RATE}, model_ms)
