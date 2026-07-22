# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the Wan2.2-owned TensorRT DiT builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import dit_builder as dit


def test_dit_builder_rejects_unqualified_profiles_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="not one of the qualified generation profiles"):
        dit.build_dit_engine("unused", profile=SimpleNamespace())
