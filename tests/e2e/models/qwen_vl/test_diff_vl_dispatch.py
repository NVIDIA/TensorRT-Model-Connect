# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned coverage for diff_vl handler dispatch."""

from __future__ import annotations

def test_qwen_vl_variants_are_owned_by_family_handler() -> None:
    from tools import diff_vl

    handler = diff_vl._find_family_diff_vl_handler("qwen2_5_vl")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith("families/qwen_vl/diff_vl.py")
    assert handler.handles_model_type("qwen_vl") is True
    assert handler.handles_model_type("Qwen2_5_VL") is True
