# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixtral FP8 quantization: checkpoint reading, FP8 experts, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import graph_ops, model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    MixtralQuantContext,
    _FP8Weight,
    calibrate_mixtral,
    dequantize_fp8,
)

VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 32, 256, 2, 4, 2, 64
EXPERTS, TOP_K, INTER = 4, 2, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

_ATTENTION = {
    "self_attn.q_proj": (ATTENTION, HIDDEN),
    "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN),
    "self_attn.o_proj": (HIDDEN, ATTENTION),
}
_EXPERT = {"w1": (INTER, HIDDEN), "w3": (INTER, HIDDEN), "w2": (HIDDEN, INTER)}


def _config_json() -> dict:
    return {
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTER,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
    }


def _write_checkpoint(path: Path, *, quantized: bool, dequantized_from=None) -> dict:
    """Write a tiny Mixtral checkpoint, ModelOpt FP8 style when ``quantized``.

    With ``dequantized_from`` (the tensors of a quantized checkpoint), the
    projections hold the values those FP8 weights stand for, in bfloat16.
    """
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }

    def add_projection(name: str, out_f: int, in_f: int, input_scale: float) -> None:
        # Draw the weight in every mode so the other tensors (router) stay identical.
        weight = torch.randn(out_f, in_f, generator=gen) * 0.1
        if dequantized_from is not None:
            scale = float(dequantized_from[f"{name}.weight_scale"])
            value = dequantized_from[f"{name}.weight"].float() * scale
            tensors[f"{name}.weight"] = value.bfloat16()
            return
        if not quantized:
            tensors[f"{name}.weight"] = weight.bfloat16()
            return
        scale = float(weight.abs().max()) / 448
        tensors[f"{name}.weight"] = (weight / scale).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale"] = torch.tensor(scale)
        tensors[f"{name}.input_scale"] = torch.tensor(input_scale)

    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.block_sparse_moe.gate.weight"] = torch.randn(
            EXPERTS, HIDDEN, generator=gen
        ).bfloat16()
        for stem, (out_f, in_f) in _ATTENTION.items():
            add_projection(f"{base}.{stem}", out_f, in_f, 0.02 if stem.endswith("o_proj") else 0.01)
        for expert in range(EXPERTS):
            for stem, (out_f, in_f) in _EXPERT.items():
                # One input scale per expert and input, shared by w1 and w3.
                scale = 0.01 * (1 + 0.1 * expert) if stem != "w2" else 0.03 * (1 + 0.1 * expert)
                add_projection(
                    f"{base}.block_sparse_moe.experts.{expert}.{stem}", out_f, in_f, scale
                )
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    return tensors


def test_calibrate_reads_attention_bytes(tmp_path: Path) -> None:
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = calibrate_mixtral(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "fp8")

    assert len(ctx.scales) == LAYERS * (len(_ATTENTION) - 1)  # o_proj stays fp16
    assert set(ctx.dense) == {f"layer.{layer}.w_o" for layer in range(LAYERS)}
    assert set(ctx.experts) == {f"layer.{layer}" for layer in range(LAYERS)}
    q = ctx.scales["layer.1.w_q"]
    assert isinstance(q, _FP8Weight) and (q.out_features, q.in_features) == (ATTENTION, HIDDEN)
    assert np.array_equal(  # bit-exact reuse
        q.packed, tensors["model.layers.1.self_attn.q_proj.weight"].view(torch.uint8).numpy()
    )
    assert q.weight_scale == pytest.approx(
        float(tensors["model.layers.1.self_attn.q_proj.weight_scale"])
    )
    assert q.input_scale == pytest.approx(0.01)
    assert ctx.should_quantize("layer.0.w_q")
    assert not ctx.should_quantize("layer.0.w_o")
    assert ctx.should_quantize("layer.1.experts.w_down")
    assert not ctx.should_quantize("layer.5.w_q")


def test_experts_keep_fp8_bytes_and_scales(tmp_path: Path) -> None:
    tensors = _write_checkpoint(tmp_path, quantized=True)
    ctx = calibrate_mixtral(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "fp8")
    layer = ctx.experts["layer.1"]

    assert len(layer.gate) == len(layer.up) == len(layer.down) == EXPERTS
    for expert in range(EXPERTS):
        for entries, stem in ((layer.gate, "w1"), (layer.up, "w3"), (layer.down, "w2")):
            name = f"model.layers.1.block_sparse_moe.experts.{expert}.{stem}"
            entry = entries[expert]
            assert np.array_equal(  # bit-exact reuse
                entry.packed, tensors[f"{name}.weight"].view(torch.uint8).numpy()
            )
            assert entry.weight_scale == pytest.approx(float(tensors[f"{name}.weight_scale"]))
            assert entry.input_scale == pytest.approx(float(tensors[f"{name}.input_scale"]))
    assert layer.gate[0].packed.shape == (INTER, HIDDEN)  # checkpoint [out, in]
    assert layer.down[0].packed.shape == (HIDDEN, INTER)
    expected = tensors["model.layers.1.block_sparse_moe.experts.0.w2.weight"].float() * float(
        tensors["model.layers.1.block_sparse_moe.experts.0.w2.weight_scale"]
    )
    assert torch.equal(dequantize_fp8(layer.down[0]), expected)


def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=False)
    with pytest.raises(ValueError, match="not an fp8 ModelOpt tensor"):
        calibrate_mixtral(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "fp8")


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        calibrate_mixtral(tmp_path, None, graph_ops, "nvfp4")


def test_calibrate_rejects_gate_up_with_different_input_scales(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=True)
    path = tmp_path / "model.safetensors"
    tensors = safetensors_torch.load_file(str(path))
    tensors["model.layers.0.block_sparse_moe.experts.2.w3.input_scale"] = torch.tensor(0.9)
    safetensors_torch.save_file(tensors, str(path))
    with pytest.raises(ValueError, match="one input scale"):
        calibrate_mixtral(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "fp8")


def test_load_weights_uses_shape_only_placeholders(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=True)
    config = ModelConfig.from_dir(tmp_path)
    ctx = calibrate_mixtral(tmp_path, config, graph_ops, "fp8")
    weights = model._MixtralModel().load_weights(
        str(tmp_path), config, precision="fp16", quant_ctx=ctx
    )

    assert weights["layer.0.w_q"].shape == (HIDDEN, ATTENTION)
    # The output projection is loaded as the dequantized fp16 weight.
    assert weights["layer.0.w_o"].shape == (ATTENTION, HIDDEN)
    assert weights["layer.0.w_o"].dtype == np.float16
    expected = dequantize_fp8(ctx.dense["layer.0.w_o"]).T.numpy().astype(np.float16)
    assert np.array_equal(weights["layer.0.w_o"], expected)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_q"].strides == (0, 0)
    assert weights["_attention_size"] == ATTENTION
    # Experts are not loaded at all; unquantized tensors still are, in fp16.
    assert "layer.0.expert.0.w_gate" not in weights
    assert weights["layer.0.router"].shape == (HIDDEN, EXPERTS)
    assert weights["layer.0.router"].dtype == np.float16
    assert weights["w_out"].shape == (HIDDEN, VOCAB)


def test_unquantized_load_weights_keeps_fp32_default(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    weights = model._MixtralModel().load_weights(str(tmp_path), config)

    assert weights["layer.0.w_q"].dtype == np.float32
    assert weights["layer.0.expert.3.w_down"].shape == (INTER, HIDDEN)


def _request(**overrides) -> SimpleNamespace:
    values = dict(
        backend="trt",
        dynamic_kv_cache=False,
        mtp_seq_len=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        tensor_parallel_size=1,
        quantization="fp8",
        fp32_layers=(),
        precision="fp16",
        max_sequence_length=16,
        verbose=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"precision": "bf16"}, "fp16 precision"),
        ({"fp32_layers": (0,)}, "fp16 precision"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1"),
        ({"quantization": "nvfp4"}, "does not support quantization"),
    ],
)
def test_build_rejects_unsupported_quantized_configs(tmp_path: Path, overrides, message) -> None:
    _write_checkpoint(tmp_path, quantized=True)
    with pytest.raises(NotImplementedError, match=message):
        model.build(_request(model_dir=tmp_path, **overrides), writer=None)


def _run_decode_step(plan, token: int) -> np.ndarray:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(bytes(plan))
    context = engine.create_execution_context()
    dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
    buffers = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = torch.int32 if engine.get_tensor_dtype(name) == trt.DataType.INT32 else None
        buffer = torch.zeros(
            shape, dtype=dtype or dtypes[engine.get_tensor_dtype(name)], device="cuda"
        )
        buffers[name] = buffer
        context.set_tensor_address(name, buffer.data_ptr())
    buffers["token_id"].fill_(token)
    buffers["attention_mask"][:, 1:] = -1e4
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return buffers["logits"].float().cpu().numpy().reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_matches_dequantized_checkpoint(tmp_path: Path) -> None:
    """The FP8 engine is small, runs, and tracks an engine built from the dequantized weights."""
    quant_dir, plain_dir = tmp_path / "quant", tmp_path / "plain"
    tensors = _write_checkpoint(quant_dir, quantized=True)
    _write_checkpoint(plain_dir, quantized=False, dequantized_from=tensors)
    instance = model._MixtralModel()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = calibrate_mixtral(quant_dir, config, graph_ops, "fp8")
    assert isinstance(ctx, MixtralQuantContext)
    weights = instance.load_weights(str(quant_dir), config, precision="fp16", quant_ctx=ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    # Expert and q/k/v weights are 8 bits instead of 32 in the plain plan; the embedding,
    # lm_head, router and o_proj stay 16 bits.
    assert len(quantized) < 0.65 * len(plain), (len(quantized), len(plain))
    reference = _run_decode_step(plain, token=3)
    actual = _run_decode_step(quantized, token=3)
    assert np.isfinite(actual).all()
    cosine = float(reference @ actual / (np.linalg.norm(reference) * np.linalg.norm(actual)))
    assert cosine > 0.95, cosine
