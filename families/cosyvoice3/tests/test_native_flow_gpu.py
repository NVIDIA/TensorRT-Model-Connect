# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tiny-network TensorRT correctness tests.

Run with COSYVOICE3_RUN_GPU_TESTS=1 in a CUDA + TensorRT + PyTorch environment.
"""

import json
import os
from dataclasses import asdict

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get("COSYVOICE3_RUN_GPU_TESTS") != "1",
    reason="Set COSYVOICE3_RUN_GPU_TESTS=1 to build and execute the native GPU test",
)]


def torch_reference(weights, cfg, x, mask, mu, t, spks, cond):
    """Straightforward PyTorch equations, independent of the TRT graph helpers."""
    import torch
    import torch.nn.functional as F
    from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

    w = {k: torch.from_numpy(v).to(x.device) for k, v in weights.items()}

    def linear(a, key):
        return F.linear(a, w[key + ".weight"], w[key + ".bias"])

    half = cfg.time_dim // 2
    frequencies = torch.exp(torch.arange(half, device=x.device) * (-np.log(10000) / (half - 1)))
    phases = 1000 * t[:, None] * frequencies[None]
    time = linear(F.silu(linear(torch.cat((phases.sin(), phases.cos()), dim=-1), "time_embed.time_mlp.0")), "time_embed.time_mlp.2")
    n = x.shape[-1]
    x = linear(torch.cat((x.transpose(1, 2), cond.transpose(1, 2), mu.transpose(1, 2),
                         spks[:, None].expand(-1, n, -1)), dim=-1), "input_embed.proj")
    position = x.transpose(1, 2)
    for i in (1, 2):
        key = f"input_embed.conv_pos_embed.conv{i}.0"
        position = F.mish(F.conv1d(F.pad(position, (cfg.conv_kernel - 1, 0)),
                                  w[key + ".weight"], w[key + ".bias"], groups=cfg.conv_groups))
    x = x + position.transpose(1, 2)
    rope, _ = RotaryEmbedding(cfg.head_dim).to(x.device).forward_from_seq_len(n)
    for i in range(cfg.depth):
        key = f"transformer_blocks.{i}"
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = linear(F.silu(time), key + ".attn_norm.linear").chunk(6, dim=-1)
        normalized = F.layer_norm(x, (cfg.dim,), eps=1e-6) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = [linear(normalized, key + ".attn.to_" + name) for name in ("q", "k", "v")]
        q, k = apply_rotary_pos_emb(q, rope), apply_rotary_pos_emb(k, rope)
        q, k, v = [a.reshape(2, n, cfg.heads, cfg.head_dim).transpose(1, 2) for a in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None].bool(), dropout_p=0)
        out = linear(out.transpose(1, 2).reshape(2, n, cfg.dim), key + ".attn.to_out.0")
        x = x + gate_a[:, None] * (out * mask.transpose(1, 2))
        norm = F.layer_norm(x, (cfg.dim,), eps=1e-6) * (1 + scale_f[:, None]) + shift_f[:, None]
        ff = linear(F.gelu(linear(norm, key + ".ff.ff.0.0"), approximate="tanh"), key + ".ff.ff.2")
        x = x + gate_f[:, None] * ff
    scale, shift = linear(F.silu(time), "norm_out.linear").chunk(2, dim=-1)
    x = F.layer_norm(x, (cfg.dim,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
    return linear(x, "proj_out").transpose(1, 2)


@pytest.fixture(scope="module")
def component(tmp_path_factory):
    import torch
    from families.cosyvoice3.checkpoint_mapper import expected_shapes
    from families.cosyvoice3.config import FlowConfig, ShapeProfile
    from families.cosyvoice3.flow_builder import build_flow_engine
    from families.cosyvoice3.flow_runtime import FlowEngine

    assert torch.cuda.is_available(), "GPU tests were explicitly requested but CUDA is unavailable"
    cfg = FlowConfig(dim=16, depth=2, heads=2, head_dim=8, ff_mult=2,
                     mel_dim=4, spk_dim=4, time_dim=8, conv_kernel=3, conv_groups=2)
    profile = ShapeProfile(4, 8, 17)
    rng = np.random.default_rng(0)
    weights = {k: rng.normal(0, .1, shape).astype(np.float32) for k, shape in expected_shapes(cfg).items()}
    plan = build_flow_engine(weights, cfg, profile, workspace_mib=64)
    path = tmp_path_factory.mktemp("cosyvoice3-flow")
    (path / "flow.plan").write_bytes(plan)
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "component": "cosyvoice3_flow_estimator", "precision": "fp32", "streaming": False,
        "architecture": asdict(cfg), "profile": asdict(profile),
    }), encoding="utf-8")
    return FlowEngine(path), weights


@pytest.mark.parametrize("frames,masked", [(4, False), (8, True), (17, False)])
def test_native_graph_parity(component, frames, masked):
    import torch
    engine, weights = component
    generator = torch.Generator(device="cuda").manual_seed(2512)
    values = {name: torch.randn((2, 4, frames), generator=generator, device="cuda") for name in ("x", "mu", "cond")}
    values["t"] = torch.tensor([0.0, 0.7], device="cuda")
    values["spks"] = torch.randn((2, 4), generator=generator, device="cuda")
    values["mask"] = torch.ones((2, 1, frames), device="cuda")
    if masked:
        values["mask"][1, :, -2:] = 0
    # Reference on CPU avoids TF32/fused GPU attention introducing another
    # implementation's precision defaults into this strict tiny-network test.
    expected = torch_reference(weights, engine.cfg, **{k: v.cpu() for k, v in values.items()})
    torch.testing.assert_close(engine(**values).cpu(), expected, atol=1e-5, rtol=1e-4)


def test_runtime_rejects_streaming_and_invalid_masks(component):
    import torch
    engine, _ = component
    values = {name: torch.zeros((2, 4, 8), device="cuda") for name in ("x", "mu", "cond")}
    values.update(t=torch.zeros(2, device="cuda"), spks=torch.zeros((2, 4), device="cuda"),
                  mask=torch.ones((2, 1, 8), device="cuda"))
    with pytest.raises(ValueError, match="offline-only"):
        engine(**values, streaming=True)
    values["mask"].zero_()
    with pytest.raises(ValueError, match="valid frame"):
        engine(**values)
