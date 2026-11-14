# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AST's Transformers model for ``classify_audio``, as tests/test_e2e.py runs it: the mono waveform through
the checkpoint's feature extractor, the raw logits of every AudioSet class with its name."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        self.spec, self.host = spec, host
        self.extractor = AutoFeatureExtractor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(ASTForAudioClassification.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        self.labels = [self.model.config.id2label[index] for index in range(len(self.model.config.id2label))]

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        rate = int(self.extractor.sampling_rate)
        audio = self.host.load_audio(str(self.host.required(request, "audio_path")), rate)
        values = self.extractor(audio, sampling_rate=rate, return_tensors="pt")["input_values"]
        values = values.to(self.spec.device, self.spec.dtype)
        logits, model_ms = self.host.timed(lambda: self.model(values).logits[0].float())
        scores = logits.cpu().tolist()
        return self.host.invocation({"classified_clips": 1, "scores": scores, "score_kind": "logit",
                                     "labels": self.labels, "top_class": max(range(len(scores)), key=scores.__getitem__)},
                                    model_ms)
