# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex's native pipeline for the trtmc-perf-serve reference backend (``speak``): the pinned official
moshi source (native_prepare.py), loaded once; per request the user audio is encoded frame by frame and the
greedy LMGen answers (at most ``max_new_tokens`` speech frames), decoded to 24 kHz audio."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trtmc_perf_serving.backends.base import BackendError, Invocation
from trtmc_perf_serving.backends.reference.common import (ReferenceSpec, invocation, load_audio, required,
                                                           tensor_observation, timed)

COMPAT = Path(__file__).resolve().parent / "tests/personaplex_audio_compat"  # the family's sphn audio shim


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        source = Path(sys.prefix) / "trtmc-reference/personaplex"
        if not (source / "moshi/moshi/offline.py").is_file():
            raise BackendError(f"the PersonaPlex reference is not prepared: {source}")
        sys.path[:0] = [str(COMPAT), str(source / "moshi"), str(source)]
        from huggingface_hub import hf_hub_download
        from moshi.models import LMGen, loaders
        from moshi.offline import warmup

        self.spec = spec
        options = {"revision": spec.revision} if spec.revision else {}
        mimi_weights = hf_hub_download(spec.model, loaders.MIMI_NAME, **options)
        model_weights = hf_hub_download(spec.model, loaders.MOSHI_NAME, **options)
        self.mimi = loaders.get_mimi(mimi_weights, spec.device)
        self.other_mimi = loaders.get_mimi(mimi_weights, spec.device)
        model = loaders.get_moshi_lm(model_weights, device=spec.device, dtype=spec.dtype).eval()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.generator = LMGen(model, audio_silence_frame_cnt=0, sample_rate=self.mimi.sample_rate, device=spec.device,
                               frame_rate=self.mimi.frame_rate, use_sampling=False, temp=0.8, temp_text=0.7, top_k=250,
                               top_k_text=25)
        for streaming in (self.mimi, self.other_mimi, self.generator):
            streaming.streaming_forever(1)
        warmup(self.mimi, self.other_mimi, self.generator, spec.device, self.frame_size)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        import torch
        from moshi.models.lm import _iterate_audio, encode_from_sphn

        frames = int(request.get("max_new_tokens", 300))
        # Mono at Mimi's rate (librosa resampling): the environment's sphn stand-in reads 24 kHz files only.
        source_audio = load_audio(str(required(request, "audio_path")), int(self.mimi.sample_rate))[None, :]
        for streaming in (self.mimi, self.other_mimi, self.generator):
            streaming.reset_streaming()

        def run() -> list[Any]:
            chunks: list[Any] = []
            for encoded in encode_from_sphn(self.mimi, _iterate_audio(source_audio, sample_interval_size=self.frame_size,
                                                                      pad=True), max_batch=1):
                for index in range(encoded.shape[-1]):
                    tokens = self.generator.step(encoded[:, :, index:index + 1])
                    if tokens is None:
                        continue
                    chunks.append(self.mimi.decode(tokens[:, 1:9])[0, 0].float())
                    self.other_mimi.decode(tokens[:, 1:9])
                    if len(chunks) >= frames:
                        return chunks
            return chunks

        chunks, model_ms = timed(run)
        if not chunks:
            raise BackendError("the native PersonaPlex produced no speech frames")
        audio = torch.cat(chunks).cpu().numpy().astype(np.float32)
        rate = int(self.mimi.sample_rate)
        return invocation({**tensor_observation(audio, artifact_base), "sample_rate": rate, "output_tokens": len(chunks),
                           "audio_seconds": audio.size / rate}, model_ms)
