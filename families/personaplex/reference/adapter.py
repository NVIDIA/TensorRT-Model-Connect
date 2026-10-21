# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex's native pipeline for the trtmc-perf-serve reference backend (``speak``): the pinned official
moshi source (reference/prepare.py), loaded once; per request the user audio is encoded frame by frame and the
greedy LMGen answers (at most ``max_new_tokens`` speech frames), decoded to 24 kHz audio."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


COMPAT = Path(__file__).resolve().parent / "tests/personaplex_audio_compat"  # the family's sphn audio shim


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        source = Path(sys.prefix) / "trtmc-reference/personaplex"
        if not (source / "moshi/moshi/offline.py").is_file():
            raise self.host.Error(f"the PersonaPlex reference is not prepared: {source}")
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
        self.system_prompt = None
        adapter_options = getattr(spec, "options", {})
        self.listen_before_answer = bool(adapter_options.get("listen_before_answer", False))
        if prompt_path := adapter_options.get("system_prompt"):
            prompt_path = Path(prompt_path)
            if not prompt_path.is_absolute():
                prompt_path = Path(sys.prefix) / prompt_path
            with np.load(prompt_path, allow_pickle=False) as prompt:
                self.system_prompt = json.loads(str(prompt["metadata"]))
                voice_embeddings = prompt["embeddings"][:self.system_prompt["voice_frames"]].copy()
            voice = Path(adapter_options["voice_prompt"])
            if not voice.is_absolute():
                voice = Path(sys.prefix) / voice
            if (self.system_prompt["precision"] != spec.precision
                    or self.system_prompt["voice"] != voice.name):
                raise self.host.Error("the reference system prompt does not match its precision or voice")
            self.generator.load_voice_prompt_embeddings(str(voice))
            # Official voice assets store BF16 embeddings; match the requested model storage dtype.
            self.generator.voice_prompt_embeddings = self.generator.voice_prompt_embeddings.to(spec.dtype)
            actual_voice = self.generator.voice_prompt_embeddings.float().cpu().numpy().reshape(-1, 4096)
            if not np.array_equal(actual_voice, voice_embeddings):
                raise self.host.Error("the reference voice embeddings differ from the prepared prefix")
            self.generator.text_prompt_tokens = self.system_prompt["text_token_ids"]
            self.generator.audio_silence_frame_cnt = self.system_prompt["silence_frames"]

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import soundfile
        import torch
        from moshi.models.lm import _iterate_audio, encode_from_sphn

        frames = int(request.get("max_new_tokens", 300))
        # Suites deliver audio at Mimi's rate (``audio_sample_rate``), so the TRTMC side reads the same samples;
        # the reference never resamples (the official sphn loader runs its resampler even on 24 kHz input).
        audio_path = str(self.host.required(request, "audio_path"))
        if (rate := soundfile.info(audio_path).samplerate) != self.mimi.sample_rate:
            raise self.host.Error(f"PersonaPlex reference input must be {self.mimi.sample_rate} Hz audio, not {rate} Hz")
        source_audio = self.host.load_audio(audio_path, int(self.mimi.sample_rate))[None, :]
        question_frames = (source_audio.shape[-1] + self.frame_size - 1) // self.frame_size
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt is not None:
            tail = int(request.get("tail_frames", 0))
            if tail < 0 or tail > frames:
                raise self.host.Error("tail_frames must fit the output frame budget")
            source_audio = np.pad(source_audio, ((0, 0), (0, tail * self.frame_size)))
        def run() -> list[Any]:
            for streaming in (self.mimi, self.other_mimi, self.generator):
                streaming.reset_streaming()
            if system_prompt is not None:
                self.generator.step_system_prompts(self.mimi)
                self.mimi.reset_streaming()
            chunks: list[Any] = []
            input_frame = 0
            for encoded in encode_from_sphn(self.mimi, _iterate_audio(source_audio, sample_interval_size=self.frame_size,
                                                                      pad=True), max_batch=1):
                for index in range(encoded.shape[-1]):
                    forced = ({"moshi_tokens": self.generator._encode_zero_frame(),
                               "text_token": self.generator.zero_text_code}
                              if getattr(self, "listen_before_answer", False) and input_frame < question_frames else {})
                    if getattr(self, "listen_before_answer", False) and input_frame == question_frames:
                        forced = {"text_token": 0}  # EPAD: official Moshi spoken-QA response trigger.
                    tokens = self.generator.step(encoded[:, :, index:index + 1], **forced)
                    input_frame += 1
                    if tokens is None:
                        continue
                    # Decode the returned stream once; the other codec is only needed for official warmup.
                    chunks.append(self.mimi.decode(tokens[:, 1:9])[0, 0].float())
                    if len(chunks) >= frames:
                        return chunks
            return chunks

        chunks, model_ms = self.host.timed(run)
        if not chunks:
            raise self.host.Error("the native PersonaPlex produced no speech frames")
        audio = torch.cat(chunks).cpu().numpy().astype(np.float32)
        rate = int(self.mimi.sample_rate)
        provenance = ({"system_prompt_voice": system_prompt["voice"],
                       "system_prompt_frames": system_prompt["frames"]}
                      if system_prompt is not None else {})
        return self.host.invocation({**self.host.tensor_observation(audio, artifact_base), "sample_rate": rate,
                                     "output_tokens": len(chunks), "audio_seconds": audio.size / rate,
                                     **provenance}, model_ms)
