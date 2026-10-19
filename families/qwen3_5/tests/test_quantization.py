# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 NVFP4 quantization: checkpoint reading, query/gate split, placeholders, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import graph_ops, model  # noqa: E402
from ..build_request import BuildRequest, coerce_request  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    Qwen35QuantContext,
    _NVFP4Weight,
    _split_query_gate,
    calibrate_qwen35,
    is_nvfp4_checkpoint,
)

# Every quantized input dimension is a multiple of the 16-element NVFP4 block.
VOCAB, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP = 32, 256, 4, 2, 64, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
DELTA_HEADS, DELTA_KV_HEADS, DELTA_DIM = 4, 2, 64
D_INNER = DELTA_HEADS * DELTA_DIM
CONV_DIM = 2 * DELTA_KV_HEADS * DELTA_DIM + D_INNER
LAYER_TYPES = ["deltanet", "attention"]
PREFIX = "model.language_model.layers"

_DELTANET = {
    "linear_attn.in_proj_qkv": (CONV_DIM, HIDDEN),
    "linear_attn.in_proj_z": (D_INNER, HIDDEN),
    "linear_attn.in_proj_a": (DELTA_HEADS, HIDDEN),
    "linear_attn.in_proj_b": (DELTA_HEADS, HIDDEN),
    "linear_attn.out_proj": (HIDDEN, D_INNER),
}
_ATTENTION = {
    "self_attn.q_proj": (HEADS * 2 * HEAD_DIM, HIDDEN),  # query and gate rows, interleaved
    "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN),
    "self_attn.o_proj": (HIDDEN, ATTENTION),
}
_MLP = {
    "mlp.gate_proj": (MLP, HIDDEN),
    "mlp.up_proj": (MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}


def _config_json() -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": len(LAYER_TYPES),
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": MLP,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": DELTA_HEADS,
            "linear_num_key_heads": DELTA_KV_HEADS,
            "linear_value_head_dim": DELTA_DIM,
            "linear_key_head_dim": DELTA_DIM,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        },
    }


def _write_checkpoint(
    path: Path, *, quantized: bool, input_scale: float = 0.02
) -> dict[str, torch.Tensor]:
    """Write a tiny hybrid Qwen3.5 checkpoint, ModelOpt NVFP4 style when ``quantized``."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.language_model.norm.weight": torch.zeros(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }

    def add_projection(name: str, out_f: int, in_f: int) -> None:
        if not quantized:
            tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
            return
        tensors[f"{name}.weight"] = torch.randint(
            0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
        )
        scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale"] = scale
        tensors[f"{name}.weight_scale_2"] = torch.tensor(0.004)
        tensors[f"{name}.input_scale"] = torch.tensor(input_scale)

    for layer, kind in enumerate(LAYER_TYPES):
        base = f"{PREFIX}.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.zeros(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.zeros(HIDDEN).bfloat16()
        if kind == "deltanet":
            mixer = _DELTANET
            tensors[f"{base}.linear_attn.A_log"] = torch.zeros(DELTA_HEADS).bfloat16()
            tensors[f"{base}.linear_attn.dt_bias"] = torch.ones(DELTA_HEADS).bfloat16()
            tensors[f"{base}.linear_attn.conv1d.weight"] = torch.randn(
                CONV_DIM, 1, 4, generator=gen
            ).bfloat16()
            tensors[f"{base}.linear_attn.norm.weight"] = torch.ones(DELTA_DIM).bfloat16()
        else:
            mixer = _ATTENTION
            tensors[f"{base}.self_attn.q_norm.weight"] = torch.zeros(HEAD_DIM).bfloat16()
            tensors[f"{base}.self_attn.k_norm.weight"] = torch.zeros(HEAD_DIM).bfloat16()
        for stem, (out_f, in_f) in {**mixer, **_MLP}.items():
            add_projection(f"{base}.{stem}", out_f, in_f)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    return tensors


def _calibrate(path: Path) -> Qwen35QuantContext:
    return calibrate_qwen35(
        path, graph_ops, "nvfp4", layer_types=LAYER_TYPES, num_heads=HEADS, head_dim=HEAD_DIM
    )


def test_calibrate_reads_every_projection_bit_exact(tmp_path: Path) -> None:
    """Every projection of both layer kinds is read, and its bytes are the checkpoint's."""
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = _calibrate(tmp_path)

    # deltanet: 5 mixer + 3 mlp; attention: q split into 2, k, v, o + 3 mlp
    assert len(ctx.scales) == 8 + 8
    qkv = ctx.scales["layer.0.deltanet_in_proj_qkv"]
    assert isinstance(qkv.weight, _NVFP4Weight)
    assert (qkv.weight.out_features, qkv.weight.in_features) == (CONV_DIM, HIDDEN)
    assert np.array_equal(
        qkv.weight.packed,
        tensors[f"{PREFIX}.0.linear_attn.in_proj_qkv.weight"].view(torch.uint8).numpy(),
    )
    assert qkv.input_scale == pytest.approx(0.02)
    assert qkv.weight.global_scale == pytest.approx(0.004)
    # tiny projection (one output per head) is still read as a quantized weight
    assert ctx.scales["layer.0.deltanet_a_proj"].weight.out_features == DELTA_HEADS
    assert ctx.should_quantize("layer.1.w_down")
    assert not ctx.should_quantize("layer.5.w_q")


def test_query_gate_split_follows_the_per_head_row_layout(tmp_path: Path) -> None:
    """q_proj rows are [head0 query, head0 gate, head1 query, ...]; each half is bit-exact."""
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = _calibrate(tmp_path)
    packed = tensors[f"{PREFIX}.1.self_attn.q_proj.weight"].view(torch.uint8).numpy()
    scale = tensors[f"{PREFIX}.1.self_attn.q_proj.weight_scale"].view(torch.uint8).numpy()

    query, gate = ctx.scales["layer.1.w_q"].weight, ctx.scales["layer.1.w_gate_attn"].weight
    assert (query.out_features, gate.out_features) == (ATTENTION, ATTENTION)
    assert query.packed.shape == (ATTENTION, HIDDEN // 2)
    for head in range(HEADS):
        first = head * 2 * HEAD_DIM
        rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        assert np.array_equal(query.packed[rows], packed[first : first + HEAD_DIM])
        assert np.array_equal(gate.packed[rows], packed[first + HEAD_DIM : first + 2 * HEAD_DIM])
        assert np.array_equal(query.block_scale_f8[rows], scale[first : first + HEAD_DIM])
        assert np.array_equal(
            gate.block_scale_f8[rows], scale[first + HEAD_DIM : first + 2 * HEAD_DIM]
        )
    with pytest.raises(ValueError, match="rows"):
        _split_query_gate(query, HEADS + 1, HEAD_DIM)


def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path) -> None:
    """A BF16 checkpoint fails with a format error instead of silently staying unquantized."""
    _write_checkpoint(tmp_path, quantized=False)
    with pytest.raises(ValueError, match="not an nvfp4 ModelOpt tensor"):
        _calibrate(tmp_path)


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Only nvfp4 is supported."""
    with pytest.raises(NotImplementedError):
        calibrate_qwen35(
            tmp_path, graph_ops, "fp8", layer_types=LAYER_TYPES, num_heads=HEADS, head_dim=HEAD_DIM
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(tmp_path: Path, bad: float) -> None:
    """A zero, negative or NaN input_scale is reported as such."""
    _write_checkpoint(tmp_path, quantized=True, input_scale=bad)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        _calibrate(tmp_path)


def test_is_nvfp4_checkpoint(tmp_path: Path) -> None:
    """Quantized checkpoints are recognised so a build without the flag can refuse them."""
    _write_checkpoint(tmp_path / "quant", quantized=True)
    _write_checkpoint(tmp_path / "plain", quantized=False)
    assert is_nvfp4_checkpoint(tmp_path / "quant")
    assert not is_nvfp4_checkpoint(tmp_path / "plain")
    assert not is_nvfp4_checkpoint(tmp_path / "missing")


def test_build_request_accepts_quantization(tmp_path: Path) -> None:
    """quantization is a real request field; 'none' and None both mean unquantized."""
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out",
        family="qwen3_5",
        task="text_generation",
        precision="fp16",
        quantization="nvfp4",
    )
    assert coerce_request(request).quantization == "nvfp4"
    assert BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out",
        family="qwen3_5",
        task="text_generation",
        precision="fp16",
    ).quantization is None


def test_load_weights_uses_shape_only_placeholders(tmp_path: Path) -> None:
    """Quantized projections get shape-only placeholders; other tensors are still loaded."""
    _write_checkpoint(tmp_path, quantized=True)
    config = ModelConfig.from_dir(tmp_path)
    ctx = _calibrate(tmp_path)
    weights = model._Qwen35Model().load_weights(str(tmp_path), config, ctx)

    assert weights["layer.0.deltanet_in_proj_qkv"].shape == (HIDDEN, CONV_DIM)
    assert weights["layer.0.deltanet_out_proj"].shape == (D_INNER, HIDDEN)
    assert weights["layer.1.w_q"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.1.w_gate_attn"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.1.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.deltanet_in_proj_qkv"].strides == (0, 0)
    assert weights["layer.1.w_o"].strides == (0, 0)
    assert weights["layer.0.conv1d_weight"].shape == (CONV_DIM, 4)
    assert weights["w_lm_head"].shape == (HIDDEN, VOCAB)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_packed_weights(tmp_path: Path) -> None:
    """The quantized engine builds and stores packed (not full-precision) weights."""
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, quantized=False)
    _write_checkpoint(quant_dir, quantized=True)
    instance = model._Qwen35Model()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = _calibrate(quant_dir)
    weights = instance.load_weights(str(quant_dir), config, ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits"} <= names
    # The projections are 4 bits instead of 16; the embedding and lm_head are not.
    assert len(quantized) < len(plain)
