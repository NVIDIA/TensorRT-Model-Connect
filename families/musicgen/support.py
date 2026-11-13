# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership of the mono T5-conditioned MusicGen medium checkpoint."""

from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    c = metadata.config
    d, t, a = (c.get(key, {}) for key in ("decoder", "text_encoder", "audio_encoder"))
    if not all(isinstance(part, dict) for part in (d, t, a)):
        return None
    if (
        c.get("model_type") == "musicgen"
        and d.get("hidden_size") == 1536
        and d.get("num_hidden_layers") == 48
        and d.get("num_codebooks") == 4
        and d.get("vocab_size") == 2048
        and t.get("model_type") == "t5"
        and t.get("d_model") == 768
        and a.get("model_type") == "encodec"
        and a.get("audio_channels") == 1
        and a.get("sampling_rate") == 32000
    ):
        return FamilySupport(("text_to_audio",), "text_to_audio", "fp32")
    return None
