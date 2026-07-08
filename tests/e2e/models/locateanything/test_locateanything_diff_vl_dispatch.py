# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything-owned diff_vl handler tests."""

from __future__ import annotations


def test_locateanything_handler_is_model_owned():
    from tools import diff_vl

    handler = diff_vl._find_family_diff_vl_handler("locateanything")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/locateanything/diff_vl.py")
    assert handler.handles_model_type("LocateAnything") is True
