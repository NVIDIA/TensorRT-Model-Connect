# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-MoE NVFP4 quantization: checkpoint reading, FP4 experts, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import graph_ops, model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    QwenMoeQuantContext,
    _NVFP4Weight,
    calibrate_qwen_moe,
    dequantize_nvfp4,
)

# Every quantized dimension is a multiple of the 16-element NVFP4 block.
VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 32, 256, 2, 4, 2, 64
EXPERTS, TOP_K, MOE_INTER = 4, 2, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

_ATTENTION = {
    "self_attn.q_proj": (ATTENTION, HIDDEN),
    "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN),
    "self_attn.o_proj": (HIDDEN, ATTENTION),
}
_EXPERT = {
    "gate_proj": (MOE_INTER, HIDDEN),
    "up_proj": (MOE_INTER, HIDDEN),
    "down_proj": (HIDDEN, MOE_INTER),
}


def _config_json(**overrides) -> dict:
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": 512,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": MOE_INTER,
        "mlp_only_layers": [],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
    }
    config.update(overrides)
    return config


def _write_checkpoint(
    path: Path, *, quantized: bool, **config_overrides
) -> dict[str, torch.Tensor]:
    """Write a tiny Qwen3-MoE checkpoint, ModelOpt NVFP4 style when ``quantized``."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }

    def add_projection(name: str, out_f: int, in_f: int, input_scale: float) -> None:
        if not quantized:
            tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
            return
        tensors[f"{name}.weight"] = torch.randint(
            0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
        )
        scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale"] = scale
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.003 + 0.001 * input_scale)
        tensors[f"{name}.input_scale"] = torch.tensor(input_scale)

    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.self_attn.q_norm.weight"] = torch.ones(HEAD_DIM).bfloat16()
        tensors[f"{base}.self_attn.k_norm.weight"] = torch.ones(HEAD_DIM).bfloat16()
        tensors[f"{base}.mlp.gate.weight"] = torch.randn(EXPERTS, HIDDEN, generator=gen).bfloat16()
        for stem, (out_f, in_f) in _ATTENTION.items():
            add_projection(f"{base}.{stem}", out_f, in_f, 0.02)
        for expert in range(EXPERTS):
            for stem, (out_f, in_f) in _EXPERT.items():
                # Different per-expert input scales: the context keeps the largest.
                add_projection(
                    f"{base}.mlp.experts.{expert}.{stem}", out_f, in_f, 0.01 * (expert + 1)
                )
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json(**config_overrides)))
    return tensors


def test_calibrate_reads_attention_bytes(tmp_path: Path) -> None:
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = calibrate_qwen_moe(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")

    assert len(ctx.scales) == LAYERS * len(_ATTENTION)
    assert set(ctx.experts) == {f"layer.{layer}" for layer in range(LAYERS)}
    q = ctx.scales["layer.1.w_q"]
    assert isinstance(q, _NVFP4Weight) and (q.out_features, q.in_features) == (ATTENTION, HIDDEN)
    assert np.array_equal(  # bit-exact reuse
        q.packed, tensors["model.layers.1.self_attn.q_proj.weight"].view(torch.uint8).numpy()
    )
    assert q.input_scale == pytest.approx(0.02)
    assert ctx.should_quantize("layer.0.w_q")
    assert ctx.should_quantize("layer.1.experts.w_down")
    assert not ctx.should_quantize("layer.5.w_q")


def test_experts_keep_fp4_bytes_and_fold_scales(tmp_path: Path) -> None:
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = calibrate_qwen_moe(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")
    layer = ctx.experts["layer.1"]

    assert layer.gate.shape == (EXPERTS, MOE_INTER, HIDDEN // 2)  # checkpoint [out, in // 2]
    assert layer.down.shape == (EXPERTS, HIDDEN, MOE_INTER // 2)
    assert layer.gate_scale.shape == (EXPERTS, MOE_INTER, HIDDEN // 16)
    assert layer.gate.dtype == np.uint8 and layer.gate_scale.dtype == np.float16
    for expert in range(EXPERTS):
        name = f"model.layers.1.mlp.experts.{expert}.down_proj"
        weight = tensors[f"{name}.weight"].view(torch.uint8).numpy()
        assert np.array_equal(layer.down[expert], weight)  # bit-exact reuse
        block = tensors[f"{name}.weight_scale"].float().numpy()
        expected = (block * float(tensors[f"{name}.weight_scale_2"])).astype(np.float16)
        assert np.array_equal(layer.down_scale[expert], expected)
    # Same values as the dequantized checkpoint weight.
    entry = _NVFP4Weight(
        layer.down[0],
        tensors["model.layers.1.mlp.experts.0.down_proj.weight_scale"].view(torch.uint8).numpy(),
        float(tensors["model.layers.1.mlp.experts.0.down_proj.weight_scale_2"]),
        0.0,
    )
    assert dequantize_nvfp4(entry).shape == (HIDDEN, MOE_INTER)


def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=False)
    with pytest.raises(ValueError, match="not an nvfp4 ModelOpt tensor"):
        calibrate_qwen_moe(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        calibrate_qwen_moe(tmp_path, None, graph_ops, "fp8")


def test_calibrate_rejects_dense_layers(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=True, mlp_only_layers=[0])
    with pytest.raises(NotImplementedError, match="dense MLP"):
        calibrate_qwen_moe(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")


def test_load_weights_uses_shape_only_placeholders(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=True)
    config = ModelConfig.from_dir(tmp_path)
    ctx = calibrate_qwen_moe(tmp_path, config, graph_ops, "nvfp4")
    weights = model._QwenMoeModel().load_weights(
        str(tmp_path), config, precision="fp16", quant_ctx=ctx
    )

    assert weights["layer.0.w_q"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.0.w_o"].shape == (ATTENTION, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_q"].strides == (0, 0)
    assert weights["_attention_size"] == ATTENTION
    # Experts are not loaded at all; unquantized tensors still are.
    assert "layer.0.experts.w_gate" not in weights
    assert weights["layer.0.router"].shape == (HIDDEN, EXPERTS)
    assert weights["w_out"].shape == (HIDDEN, VOCAB)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_packed_weights(tmp_path: Path) -> None:
    """The quantized engine builds and stores packed (not full-precision) weights."""
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, quantized=False)
    _write_checkpoint(quant_dir, quantized=True)
    instance = model._QwenMoeModel()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config, precision="fp16"), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = calibrate_qwen_moe(quant_dir, config, graph_ops, "nvfp4")
    assert isinstance(ctx, QwenMoeQuantContext)
    weights = instance.load_weights(str(quant_dir), config, precision="fp16", quant_ctx=ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # Attention and expert weights are 4 bits instead of 16.
    # A plan that stored the experts at 16 bits again would miss this bound.
    assert len(quantized) < 0.6 * len(plain), (len(quantized), len(plain))
