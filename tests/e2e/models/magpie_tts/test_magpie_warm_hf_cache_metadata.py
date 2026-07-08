# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.families import (
    family_hf_warm_dependencies,
    family_hf_warm_files,
)


def test_magpie_reference_dependencies_are_family_owned() -> None:
    deps = dict(family_hf_warm_dependencies("magpie_tts"))

    assert deps["tts-asr-verifier"] == "openai/whisper-large-v3-turbo"
    assert deps["magpie-nanocodec"] == (
        "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
    )
    assert deps["magpie-byt5-tokenizer"] == "google/byt5-small"
    assert deps["magpie-wavlm-discriminator"] == "microsoft/wavlm-base-plus"


def test_magpie_speaker_encoder_checkpoint_is_warmed_as_an_exact_file() -> None:
    assert family_hf_warm_files("magpie_tts") == [
        (
            "magpie-speaker-encoder",
            "Edresson/Speaker_Encoder_H_ASP",
            "pytorch_model.bin",
        )
    ]
