# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Gemma generations this family will build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from families.gemma.model import _SUPPORTED_MODEL_TYPES, build as build_family
    from families.gemma.support import describe
    from tensorrt_model_connect import BuildRequest
    from tensorrt_model_connect.model_support import ModelMetadata
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _model_dir(tmp_path: Path, model_type: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "vocab_size": 32,
                "max_position_embeddings": 256,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _build(model_dir: Path) -> None:
    build_family(
        BuildRequest(
            model_dir=model_dir,
            output_path=model_dir / "out.bundle",
            family="gemma",
            task="text_generation",
            precision="fp16",
        ),
        writer=None,
    )


def test_the_gate_matches_what_support_claims() -> None:
    """The builder and the identity check must name the same generations."""
    for model_type in _SUPPORTED_MODEL_TYPES:
        assert (
            describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is not None
        )


def test_a_later_gemma_generation_is_refused(tmp_path: Path) -> None:
    """Gemma 3n and 4 need machinery this family still does not have.

    Gemma 4 adds vision and audio towers, per-layer input embeddings and
    KV-shared layers; Gemma 3n is its own architecture again. Neither is built
    here, so a prefix check would let them build a full-attention graph and
    generate quietly wrong text. The refusal names the type so the message is
    actionable. Gemma 3 text is supported and is covered below.
    """
    for model_type in ("gemma3n", "gemma4", "gemma4_text"):
        directory = _model_dir(tmp_path / model_type.replace("_", ""), model_type)
        with pytest.raises(ValueError, match=re.escape(f"model_type={model_type!r}")):
            _build(directory)


def test_an_unrelated_model_type_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path / "llama", "llama")
    with pytest.raises(ValueError, match="does not support model_type"):
        _build(directory)


def test_the_supported_generations_pass_the_gate(tmp_path: Path) -> None:
    """gemma and gemma2 must get past the type check and fail later instead.

    The directory holds no weights, so the build cannot finish; what matters is
    that it stops for a reason other than the model type.
    """
    for model_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        directory = _model_dir(tmp_path / model_type, model_type)
        with pytest.raises(Exception) as caught:  # noqa: PT011 - any later failure will do
            _build(directory)
        assert "does not support model_type" not in str(caught.value)


def test_gemma3_refuses_fp16(tmp_path: Path) -> None:
    """Gemma 3 activations exceed the fp16 range, so fp16 is refused.

    Measured on the reference in fp32, largest absolute value leaving a decoder
    layer against the fp16 maximum of 65504: gemma-3-270m peaks at 102956 and
    gemma-3-4b at 298680, both of which overflow and make the engine emit token
    0 repeatedly. gemma-3-1b peaks at 61040, inside the range by 7%, which is
    luck rather than headroom.
    """
    for model_type in ("gemma3", "gemma3_text"):
        directory = _model_dir(tmp_path / f"fp16-{model_type}", model_type)
        with pytest.raises(NotImplementedError, match="does not support fp16"):
            _build_with(directory, precision="fp16")


def test_gemma2_keeps_fp16(tmp_path: Path) -> None:
    """Gemma 2 peaks at 4060, sixteen times inside the fp16 range."""
    directory = _model_dir(tmp_path / "fp16-gemma2", "gemma2")
    with pytest.raises(Exception) as caught:  # noqa: PT011 - a later failure is fine
        _build_with(directory, precision="fp16")
    assert "does not support fp16" not in str(caught.value)


def _build_with(model_dir: Path, *, precision: str) -> None:
    build_family(
        BuildRequest(
            model_dir=model_dir,
            output_path=model_dir / "out.bundle",
            family="gemma",
            task="text_generation",
            precision=precision,
        ),
        writer=None,
    )
