# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import plugin modules")

from tensorrt_model_connect.families import find_plugin


def _plugin(model_type: str):
    plugin = find_plugin(model_type)
    assert plugin is not None
    return plugin

def test_runtime_strategy() -> None:
    plugin = _plugin("phi")
    assert getattr(plugin, "runtime_strategy", None) == "phi_decoder_kv_cache"



def test_phi_rejects_phi_moe_alias() -> None:
    plugin = _plugin("phi")
    assert plugin.name == "phi"
    assert not plugin.matches("phimoe")
