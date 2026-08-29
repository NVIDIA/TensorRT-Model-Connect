# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 native TensorRT MoE layer availability policy.

The policy decides whether the DeepSeek-V2 MoE block emits ``INetworkDefinition.add_moe()``
or its portable per-expert subgraph. TensorRT rejects ``addMoE`` on SM 12.x
inside the call itself, so the decision has to be correct before the layer is
added; these tests pin that decision without requiring a GPU.
"""

from __future__ import annotations

import pytest

# Import the submodule directly: the family package proxies attribute access
# to plugin.py, which would pull in the full builder and its dependencies.
import tensorrt_model_connect.families.deepseek_v2.native_moe as native_moe


@pytest.mark.parametrize("compute_capability", [(10, 0), (10, 3)])
def test_native_moe_enabled_on_supported_datacenter_blackwell(compute_capability):
    assert native_moe.native_moe_enabled_for(
        compute_capability, layer_available=True) is True


@pytest.mark.parametrize(
    "compute_capability",
    [
        (12, 0),  # GeForce / RTX PRO Blackwell: TensorRT rejects addMoE
        (11, 0),  # Thor: permitted by TensorRT but not qualified here
        (9, 0),   # Hopper
        (8, 9),   # Ada
        (8, 0),   # Ampere
    ],
)
def test_native_moe_disabled_off_supported_architectures(compute_capability):
    assert native_moe.native_moe_enabled_for(
        compute_capability, layer_available=True) is False


def test_native_moe_disabled_when_layer_missing():
    """An older TensorRT without IMoELayer keeps the per-expert path."""
    assert native_moe.native_moe_enabled_for(
        (10, 0), layer_available=False) is False


def test_native_moe_disabled_when_capability_unknown():
    """An unusable or absent CUDA device must not select the native path."""
    assert native_moe.native_moe_enabled_for(
        None, layer_available=True) is False


def test_native_moe_disabled_by_opt_out():
    """The escape hatch forces the per-expert path on a supported GPU."""
    assert native_moe.native_moe_enabled_for(
        (10, 0), layer_available=True, disabled=True) is False


def test_use_native_moe_honours_disable_env(monkeypatch):
    monkeypatch.setenv(native_moe.DISABLE_ENV, "1")
    monkeypatch.setattr(
        native_moe, "current_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(native_moe, "native_moe_layer_available", lambda: True)
    assert native_moe.use_native_moe() is False


def test_use_native_moe_selects_native_on_sm100(monkeypatch):
    monkeypatch.delenv(native_moe.DISABLE_ENV, raising=False)
    monkeypatch.setattr(
        native_moe, "current_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(native_moe, "native_moe_layer_available", lambda: True)
    assert native_moe.use_native_moe() is True


def test_current_compute_capability_survives_missing_cuda(monkeypatch):
    """No CUDA runtime is a normal outcome, not an exception."""
    monkeypatch.setattr(native_moe, "_cuda_runtime", lambda: None)
    assert native_moe.current_compute_capability() is None
