# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-owned diff_logits handler tests."""

from __future__ import annotations

def test_rwkv_handler_is_model_owned():
    from tools import diff_logits

    handler = diff_logits._find_family_diff_logits_handler("rwkv")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/rwkv/diff_logits.py")
    assert handler.handles_model_type("RWKV") is True
