# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model entry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="model entry imports TensorRT builders")

from tensorrt_model_connect.models.phi import model

def test_runtime_strategy() -> None:
    assert model.runtime_strategy == "phi_decoder_kv_cache"

def test_neighbor_alias_is_rejected() -> None:
    assert not model.matches("phimoe")
