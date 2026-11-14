# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-width BF16 RoPE must materialize the official product rounding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.diffusion_gemma.tests.test_graph_reference import _numpy, _reference_dependencies, _run


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("layer_index", [0, 5])
def test_rope_matches_official_bfloat16_products_exactly(layer_index):
    torch = _reference_dependencies()
    import ml_dtypes
    import tensorrt as trt
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaTextRotaryEmbedding, apply_rotary_pos_emb,
    )
    from families.diffusion_gemma.tests.original_reference import _assert_official_source
    from families.diffusion_gemma.graph import Graph, _create, _mark, _rope, _serialize, rope_tables

    config = json.loads((Path(__file__).parent / "data/diffusiongemma-26b-a4b-config.json").read_text())["text_config"]
    full = config["layer_types"][layer_index] == "full_attention"
    width = config["global_head_dim"] if full else config["head_dim"]
    heads = config["num_global_key_value_heads"] if full else config["num_key_value_heads"]
    length = 23
    values = np.random.default_rng(112).normal(size=(length, heads, width)).astype(ml_dtypes.bfloat16)
    positions = np.arange(length, dtype=np.int32)
    tables = rope_tables(config, layer_index, 32)
    _assert_official_source()
    rotary = DiffusionGemmaTextRotaryEmbedding(DiffusionGemmaTextConfig(**config), device="cuda")
    value = torch.tensor(values.astype(np.float32), dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    position_tensor = torch.tensor(positions, device="cuda").unsqueeze(0)
    kind = config["layer_types"][layer_index]
    cos, sin = rotary(value, position_tensor, kind)
    cosine = cos.squeeze(0).float().cpu().numpy()
    sine = sin.squeeze(0).float().cpu().numpy()
    # These are the BF16 constants actually consumed by this network. Derive
    # their expected values independently, including the reference frequencies.
    np.testing.assert_array_equal(tables[0][:length].astype(ml_dtypes.bfloat16).astype(np.float32), cosine)
    np.testing.assert_array_equal(tables[1][:length].astype(ml_dtypes.bfloat16).astype(np.float32), sine)
    expected = apply_rotary_pos_emb(value, cos, sin, unsqueeze_dim=2).squeeze(0).float().cpu().numpy()
    # This fixture contains cancellation that distinguishes the observed
    # multiply/add contraction from the official two rounded BF16 products.
    wide = values.astype(np.float32)
    rotated = np.concatenate([-wide[..., width // 2:], wide[..., :width // 2]], axis=-1)
    left = (wide * cosine[:, None]).astype(ml_dtypes.bfloat16).astype(np.float32)
    contracted = (left + rotated * sine[:, None]).astype(ml_dtypes.bfloat16).astype(np.float32)
    assert np.any(contracted != expected)

    logger, builder, network, settings = _create("bf16", False)
    graph = Graph(network, config, {}, "bf16")
    source = network.add_input("value", trt.bfloat16, values.shape)
    position_ids = network.add_input("positions", trt.int32, positions.shape)
    _mark(graph, _rope(graph, source, position_ids, tables), "rotated")
    plan = _serialize(builder, network, settings, builder.create_optimization_profile(), graph.buffers)
    actual = _run(plan, {"value": _numpy(value.squeeze(0), "bf16"), "positions": positions})["rotated"]
    # No matmul or reduction participates: the specified BF16 rounding is exact.
    np.testing.assert_array_equal(actual.astype(np.float32), expected)
