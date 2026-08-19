# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for SegFormer's checkpoint-owned numerical contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("tensorrt", reason="TensorRT is required for SegFormer graph tests")

from tensorrt_model_connect.families.segformer import graph_ops  # noqa: E402


def test_numerical_contract_preserves_checkpoint_eps_and_exact_gelu() -> None:
    config = SimpleNamespace(rms_norm_eps=1e-6, hidden_act="gelu")

    layer_norm_eps, hidden_act = graph_ops.resolve_numerical_contract(config)

    assert layer_norm_eps == 1e-6
    assert hidden_act == "gelu"


def test_numerical_contract_rejects_unsupported_activation() -> None:
    config = SimpleNamespace(rms_norm_eps=1e-6, hidden_act="not-gelu")

    with pytest.raises(ValueError, match="Unsupported SegFormer activation"):
        graph_ops.resolve_numerical_contract(config)


def test_gelu_dispatch_keeps_exact_and_tanh_variants_distinct(monkeypatch) -> None:
    calls = []
    exact = object()
    approximate = object()

    monkeypatch.setattr(
        graph_ops,
        "add_gelu_erf",
        lambda _network, _inp, dtype=np.float32: calls.append(("erf", dtype)) or exact,
    )
    monkeypatch.setattr(
        graph_ops,
        "add_gelu_new",
        lambda _network, _inp, dtype=np.float32: calls.append(("tanh", dtype))
        or approximate,
    )

    assert graph_ops.add_activation(None, None, "gelu") is exact
    assert graph_ops.add_activation(None, None, "gelu_new") is approximate
    assert calls == [("erf", np.float32), ("tanh", np.float32)]


def test_decode_head_boundary_casts_fp16_features_to_fp32() -> None:
    fp16_feature = SimpleNamespace(dtype=graph_ops.trt.float16)
    fp32_feature = SimpleNamespace(dtype=graph_ops.trt.float32)
    calls = []

    class CastLayer:
        def get_output(self, index):
            assert index == 0
            return fp32_feature

    class Network:
        def add_cast(self, tensor, dtype):
            calls.append((tensor, dtype))
            return CastLayer()

    network = Network()

    assert graph_ops.begin_fp32_decode_head(network, fp16_feature) is fp32_feature
    assert calls == [(fp16_feature, graph_ops.trt.float32)]
    assert graph_ops.begin_fp32_decode_head(network, fp32_feature) is fp32_feature
    assert len(calls) == 1
