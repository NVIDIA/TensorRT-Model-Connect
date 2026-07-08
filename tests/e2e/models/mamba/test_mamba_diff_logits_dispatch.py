# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba-owned diff_logits handler tests."""

from __future__ import annotations

def test_mamba_handler_is_model_owned():
    from tools import diff_logits

    handler = diff_logits._find_family_diff_logits_handler("mamba")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/mamba/diff_logits.py")
    assert handler.handles_model_type("Mamba") is True
