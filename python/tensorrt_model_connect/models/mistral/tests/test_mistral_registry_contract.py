# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model entry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="model entry imports TensorRT builders")

from tensorrt_model_connect.models.mistral import model

def test_runtime_strategy() -> None:
    assert model.runtime_strategy == "mistral_decoder_kv_cache"

def test_no_embed_input() -> None:
    assert not getattr(model, "embed_input", False)
