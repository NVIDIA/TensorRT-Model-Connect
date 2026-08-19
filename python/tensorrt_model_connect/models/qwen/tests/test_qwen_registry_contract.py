# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model entry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="model entry imports TensorRT builders")

from tensorrt_model_connect.models.qwen import model

def test_runtime_strategy() -> None:
    assert model.runtime_strategy == "qwen_decoder_kv_cache"

def test_no_embed_input() -> None:
    assert not getattr(model, "embed_input", False)

def test_owned_alias_matches() -> None:
    assert model.matches("qwen3")

def test_neighbor_alias_is_rejected() -> None:
    assert not model.matches("qwen2_vl")
