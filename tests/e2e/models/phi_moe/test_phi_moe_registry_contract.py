# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model entry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="model entry imports TensorRT builders")

from tensorrt_model_connect.families.phi_moe import model


def test_owned_alias_matches() -> None:
    assert model.matches("phimoe")

def test_neighbor_alias_is_rejected() -> None:
    assert not model.matches("phi3")
