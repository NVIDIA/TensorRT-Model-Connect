# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART decoder masking and activation contracts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from families.bart import graph_ops, model


def test_decoder_cross_attention_uses_source_padding_mask(monkeypatch) -> None:
    class FakeLayer:
        def __init__(self, output):
            self.output = output
            self.reshape_dims = None
            self.axis = None

        def get_output(self, index):
            assert index == 0
            return self.output

    class FakeNetwork:
        def add_shuffle(self, tensor):
            return FakeLayer(("shuffle", tensor))

        def add_concatenation(self, tensors):
            return FakeLayer(("concatenation", tuple(tensors)))

        def add_elementwise(self, left, right, operation):
            return FakeLayer(("elementwise", left, right, operation))

    def passthrough(_network, tensor, *_args, **_kwargs):
        return tensor

    attention_masks = []

    def capture_attention(_network, query, _key, _value, *, mask=None, **_kwargs):
        attention_masks.append(mask)
        return query

    monkeypatch.setattr(graph_ops, "add_matmul_rhs_constant", passthrough)
    monkeypatch.setattr(graph_ops, "add_bias_sum", passthrough)
    monkeypatch.setattr(graph_ops, "add_layer_norm_native", passthrough)
    monkeypatch.setattr(graph_ops, "add_activation", passthrough)
    monkeypatch.setattr(graph_ops, "add_attention_from_rows", capture_attention)

    self_attention_mask = object()
    cross_attention_mask = object()
    model._add_bart_decoder_layer(
        network=FakeNetwork(),
        hidden=object(),
        cache_k=object(),
        cache_v=object(),
        cross_k=object(),
        cross_v=object(),
        attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        eps=1e-5,
        weights=defaultdict(lambda: np.zeros(1, dtype=np.float32)),
        prefix="layer.0",
        hidden_size=16,
        num_heads=4,
        head_dim=4,
        ffn_dim=32,
        max_cache_length=8,
        max_enc_seq=8,
    )

    assert attention_masks == [
        ("shuffle", self_attention_mask),
        ("shuffle", cross_attention_mask),
    ]


def test_bart_gelu_dispatch_matches_checkpoint_variants(monkeypatch) -> None:
    exact = object()
    approximate = object()
    monkeypatch.setattr(graph_ops, "add_gelu_erf", lambda *_args, **_kwargs: exact)
    monkeypatch.setattr(graph_ops, "add_gelu_new", lambda *_args, **_kwargs: approximate)

    assert graph_ops.add_activation(None, object(), "gelu") is exact
    assert graph_ops.add_activation(None, object(), "gelu_new") is approximate
    assert graph_ops.add_activation(None, object(), "gelu_pytorch_tanh") is approximate
