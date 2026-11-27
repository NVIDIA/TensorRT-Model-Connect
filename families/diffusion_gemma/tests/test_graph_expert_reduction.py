# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discriminate grouped expert reduction from BF16 per-expert accumulation."""

from __future__ import annotations

import numpy as np
import pytest

from families.diffusion_gemma.tests.test_graph_reference import (
    _numpy,
    _reference_dependencies,
    _run,
    _tiny_config,
)


# Preserve the existing BF16 layer oracle's numerical gates.
ATOL = RTOL = 0.006


def _fixture(torch, implementation):
    modeling = pytest.importorskip("transformers.models.diffusion_gemma.modeling_diffusion_gemma")
    config = _tiny_config()
    config.num_experts = config.top_k_experts = 8
    config._experts_implementation = implementation
    experts = modeling.DiffusionGemmaTextExperts(config).to(dtype=torch.bfloat16)
    pre = modeling.DiffusionGemmaRMSNorm(16, eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
    post = modeling.DiffusionGemmaRMSNorm(16, eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
    with torch.no_grad():
        experts.gate_up_proj.zero_()
        experts.down_proj.zero_()
        # Only one intermediate feature is active. The two large opposite
        # experts make early BF16 rounding discard meaningful small terms.
        experts.gate_up_proj[:, 0, 0] = 0.25
        experts.gate_up_proj[:, config.moe_intermediate_size, 0] = 0.25
        for expert, scale in enumerate((4096.0, 16.0, -4096.0, 16.0)):
            experts.down_proj[expert, 0, 0] = scale
        experts.down_proj[7, 1, 0] = 16.0
    hidden = torch.zeros((1, 16), dtype=torch.bfloat16)
    hidden[0, 0] = 4.0
    indices = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    coefficients = torch.full((1, 8), 0.125, dtype=torch.float32)
    return config, experts, pre, post, hidden, indices, coefficients


def _grouped_reduction_formula(torch, experts, hidden, coefficients):
    # Compute each original expert's BF16 projections, then apply the
    # independent official grouped implementation's FP32 weighted reduction.
    terms = []
    for index in range(8):
        gate, up = torch.nn.functional.linear(hidden, experts.gate_up_proj[index]).chunk(2, dim=-1)
        value = torch.nn.functional.linear(experts.act_fn(gate) * up, experts.down_proj[index])
        terms.append(value.float() * coefficients[:, index:index + 1])
    return torch.stack(terms).sum(dim=0).to(torch.bfloat16)


def _assert_same(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_eight_experts_reject_early_bfloat16_accumulation():
    torch = pytest.importorskip("torch")
    _, experts, pre, post, hidden, indices, coefficients = _fixture(torch, "eager")
    with torch.inference_mode():
        normalized = pre(hidden)
        old = experts(normalized, indices, coefficients)
        grouped = _grouped_reduction_formula(torch, experts, normalized, coefficients)
        assert old[0, 0].item() == 3.6875
        assert grouped[0, 0].item() == 3.359375
        with pytest.raises(AssertionError):
            _assert_same(post(old).float().numpy(), post(grouped).float().numpy())


@pytest.mark.gpu
@pytest.mark.trt
def test_graph_matches_official_grouped_experts_under_cancellation():
    torch = _reference_dependencies()
    from families.diffusion_gemma.graph import Graph, _create, _experts, _mark, _serialize

    config, experts, pre, post, hidden, indices, coefficients = _fixture(torch, "grouped_mm")
    experts, pre, post = experts.cuda(), pre.cuda(), post.cuda()
    hidden, indices, coefficients = hidden.cuda(), indices.cuda(), coefficients.cuda()
    with torch.inference_mode():
        expected = post(experts(pre(hidden), indices, coefficients))
    weights = {
        "experts.gate_up_proj": _numpy(experts.gate_up_proj, "bf16"),
        "experts.down_proj": _numpy(experts.down_proj, "bf16"),
        "pre_feedforward_layernorm_2.weight": _numpy(pre.weight, "bf16"),
        "post_feedforward_layernorm_2.weight": _numpy(post.weight, "bf16"),
        "router.scale": np.ones(16, dtype=np.float32),
        "router.per_expert_scale": np.ones(8, dtype=np.float32),
        "router.proj.weight": np.zeros((8, 16), dtype=np.float32),
    }
    logger, builder, network, settings = _create("bf16", False)
    graph = Graph(network, config.to_dict(), weights, "bf16")
    value = network.add_input("hidden_states", graph.dtype, (1, 16))
    _mark(graph, _experts(graph, value), "routed")
    plan = _serialize(builder, network, settings, builder.create_optimization_profile(), graph.buffers)
    actual = _run(plan, {"hidden_states": _numpy(hidden, "bf16")})["routed"]
    _assert_same(actual.astype(np.float32), expected.float().cpu().numpy())
