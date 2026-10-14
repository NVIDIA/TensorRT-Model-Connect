# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny-random parity: Gemma 4 text tower and LTX2TextConnectors vs transformers / diffusers.

No checkpoint is needed: random modules are built from shrunk configs that keep every
feature of the real LTX-2.5 configs (sliding + full layers, head_dim 2x on full layers,
one global KV head with k_eq_v, proportional partial RoPE, a sliding window shorter than
the sequence, left padding, per-modality projections, registers, gated split-RoPE
connectors), saved as safetensors and built with the family builders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for LTX-2.5 engine parity tests", allow_module_level=True)
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.ltx2.tests.engine_runner import cosine, rel_l2, run_plan  # noqa: E402

SEQ = 16
PAD = 5

TINY_GEMMA = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 160,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "sliding_window": 8,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "enable_moe_block": False,
    "use_double_wide_mlp": False,
    "attention_bias": False,
    "pad_token_id": 0,
}

TINY_CONNECTORS = {
    "caption_channels": 64,
    "text_proj_in_factor": 7,
    "video_connector_num_attention_heads": 4,
    "video_connector_attention_head_dim": 16,
    "video_connector_num_layers": 2,
    "video_connector_num_learnable_registers": 8,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 4,
    "audio_connector_attention_head_dim": 8,
    "audio_connector_num_layers": 2,
    "audio_connector_num_learnable_registers": 8,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 64,
    "audio_hidden_dim": 32,
    "proj_bias": True,
    "causal_temporal_positioning": False,
}


def _randomize(module: "torch.nn.Module", seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name or name.endswith("_norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[-1] ** 0.5)
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
        for name, b in module.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand(b.shape, generator=gen))


def _ids_and_mask(vocab: int):
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(3, vocab, (1, SEQ), generator=gen, dtype=torch.int64)
    ids[:, :PAD] = 0
    mask = torch.ones(1, SEQ, dtype=torch.int64)
    mask[:, :PAD] = 0
    return ids, mask


def _tiny_gemma(tmp_path: Path):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel

    extra = {"global_head_dim", "num_global_key_value_heads"}
    cfg = Gemma4UnifiedTextConfig(**{k: v for k, v in TINY_GEMMA.items()
                                     if k in extra or hasattr(Gemma4UnifiedTextConfig, k)})
    cfg._attn_implementation = "eager"
    model = Gemma4UnifiedTextModel(cfg).eval()
    _randomize(model, 11)
    folder = tmp_path / "text_encoder"
    folder.mkdir()
    state = {f"model.language_model.{k}": v.to(torch.bfloat16).contiguous()
             for k, v in model.state_dict().items() if "rotary_emb" not in k}
    safetensors_torch.save_file(state, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(json.dumps({"text_config": TINY_GEMMA}), encoding="utf-8")
    return model, folder


def _gemma_reference(model, ids, mask, dtype):
    model = model.to("cuda", dtype)
    with torch.no_grad():
        out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True)
    return torch.stack(out.hidden_states, dim=-1).flatten(2, 3).float().cpu()


def test_gemma4_tiny_parity(tmp_path: Path) -> None:
    from families.ltx2.text_encoder_builder import build_gemma_engine

    model, folder = _tiny_gemma(tmp_path)
    ids, mask = _ids_and_mask(TINY_GEMMA["vocab_size"])
    plan = build_gemma_engine(folder, seq_len=SEQ)
    got = run_plan(plan, {"input_ids": ids.int(), "attention_mask": mask.int()})["packed"].float().cpu()
    ref32 = _gemma_reference(model, ids, mask, torch.float32)
    ref16 = _gemma_reference(model, ids, mask, torch.bfloat16)
    valid = slice(PAD, SEQ)  # padding rows are zeroed by the connectors and never compared
    assert torch.isfinite(got).all(), "padding rows must stay finite (they feed the connectors' select)"
    c32 = cosine(got[:, valid], ref32[:, valid])
    c16 = cosine(got[:, valid], ref16[:, valid])
    base = cosine(ref16[:, valid], ref32[:, valid])
    print(f"gemma4 tiny: cos vs fp32 {c32:.6f}, vs bf16 {c16:.6f}, relL2 fp32 "
          f"{rel_l2(got[:, valid], ref32[:, valid]):.4e}; torch bf16 vs fp32 cos {base:.6f} "
          f"relL2 {rel_l2(ref16[:, valid], ref32[:, valid]):.4e}")
    n = TINY_GEMMA["num_hidden_layers"] + 1
    per_layer = [cosine(got[:, valid].view(-1, TINY_GEMMA["hidden_size"], n)[..., i],
                        ref32[:, valid].view(-1, TINY_GEMMA["hidden_size"], n)[..., i]) for i in range(n)]
    print("gemma4 tiny per-state cos vs fp32:", " ".join(f"{c:.5f}" for c in per_layer))
    assert c32 > 0.999
    assert c16 > 0.999


def test_connectors_tiny_parity(tmp_path: Path) -> None:
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    from families.ltx2.text_encoder_builder import build_connectors_engine

    conn = LTX2TextConnectors(**TINY_CONNECTORS).eval()
    _randomize(conn, 23)
    folder = tmp_path / "connectors"
    folder.mkdir()
    safetensors_torch.save_file({k: v.contiguous() for k, v in conn.state_dict().items()},
                                str(folder / "diffusion_pytorch_model.safetensors"))
    (folder / "config.json").write_text(json.dumps(TINY_CONNECTORS), encoding="utf-8")
    width = TINY_CONNECTORS["caption_channels"] * TINY_CONNECTORS["text_proj_in_factor"]
    packed = (3.0 * torch.randn(1, SEQ, width, generator=torch.Generator().manual_seed(3))).to(torch.bfloat16)
    _, mask = _ids_and_mask(16)
    plan = build_connectors_engine(folder, seq_len=SEQ)
    got = run_plan(plan, {"packed": packed, "attention_mask": mask.int()})
    for dtype in (torch.float32, torch.bfloat16):
        ref = conn.to("cuda", dtype)
        with torch.no_grad():
            v, a, m = ref(packed.cuda().to(dtype), mask.cuda())
        assert bool((m == 1).all())
        cv = cosine(got["video_context"].float().cpu(), v.float().cpu())
        ca = cosine(got["audio_context"].float().cpu(), a.float().cpu())
        print(f"connectors tiny ({dtype}): video cos {cv:.6f}, audio cos {ca:.6f}")
        assert cv > 0.999
        assert ca > 0.999
