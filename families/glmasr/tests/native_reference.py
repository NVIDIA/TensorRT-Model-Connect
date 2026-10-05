# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR's native pipeline for the trtmc-perf-serve reference backend (``transcribe``).

The checkpoint is a speech-conditioned causal LM, not a speech-seq2seq model, so the generic speech
adapter cannot load it: the audio and the transcription instruction go through the processor's chat
template, and the transcript is the greedy continuation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


INSTRUCTION = "Please transcribe this audio into text"
SAMPLE_RATE = 16_000


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        from transformers import GlmAsrForConditionalGeneration, GlmAsrProcessor

        self.spec = spec
        self.processor = GlmAsrProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = GlmAsrForConditionalGeneration.from_pretrained(
            spec.model, dtype=spec.dtype, **spec.pretrained_kwargs()).to(spec.device).eval()

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        audio = self.host.load_audio(str(self.host.required(request, "audio_path")), SAMPLE_RATE)
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio},
                                                     {"type": "text", "text": INSTRUCTION}]}]
        inputs = self.processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True,
                                                    sampling_rate=SAMPLE_RATE).to(self.spec.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.spec.dtype)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated, model_ms = self.host.timed(lambda: self.model.generate(
            **inputs, max_new_tokens=int(request.get("max_new_tokens", 128)), do_sample=False))
        token_ids = [int(token) for token in generated[0, prompt_tokens:].tolist()]
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        seconds = len(audio) / SAMPLE_RATE
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids),
                                     "input_audio_seconds": seconds}, model_ms,
                                    realtime_factor=seconds / (model_ms / 1000.0))
