# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import model modules")

from tensorrt_model_connect.models import find_model


def _model(model_type: str):
    model = find_model(model_type)
    assert model is not None
    return model


def test_runtime_strategy() -> None:
    model = _model("nemotron_speech_streaming")
    assert (
        getattr(model, "runtime_strategy", None) == "nemotron_speech_streaming_speech_to_text_rnnt"
    )


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    (
        ("nemotron_asr_streaming", "NemotronAsrStreamingForRNNT"),
        ("nemotron3_5_asr", "Nemotron3_5AsrForRNNT"),
    ),
)
def test_real_hf_metadata_resolves_to_streaming_family(
    model_type: str,
    architecture: str,
) -> None:
    config = SimpleNamespace(
        model_type=model_type,
        architectures=[architecture],
        raw={
            "model_type": model_type,
            "architectures": [architecture],
        },
    )

    model = find_model(config)

    assert model is not None
    assert model.name == "nemotron_speech_streaming"


def test_manifest_aliases_and_model_matcher_are_aligned() -> None:
    import tomllib

    family = Path(__file__).resolve().parents[5] / (
        "python/tensorrt_model_connect/models/nemotron_speech_streaming"
    )
    metadata = tomllib.loads((family / "MODEL.toml").read_text(encoding="utf-8"))
    model = _model("nemotron_speech_streaming")
    accepted = {
        "enc_dec_rnnt_bpe",
        "enc_dec_rnnt_bpe_with_prompt",
        "fastconformer_cacheaware_rnnt",
        "nemotron3_5_asr",
        "nemotron_3_5_asr_streaming",
        "nemotron_asr_streaming",
        "nemotron_speech_streaming",
        "nemotron_speech_streaming_rnnt",
        "nemotronspeechstreaming",
        "rnnt_bpe",
    }

    assert accepted == set(metadata["aliases"])
    assert all(model.matches(alias) for alias in metadata["aliases"])
    assert set(metadata["architecture_patterns"]) == {
        "Nemotron3_5AsrForRNNT",
        "NemotronAsrStreamingForRNNT",
    }
