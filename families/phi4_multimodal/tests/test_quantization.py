# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-4-multimodal FP8 / NVFP4 quantization: checkpoint reading, weight loading, refusals, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorrt_model_connect import BuildRequest  # noqa: E402

from families.phi4_multimodal import graph_ops, model  # noqa: E402
from families.phi4_multimodal.config import ModelConfig  # noqa: E402
from families.phi4_multimodal.quantization import (  # noqa: E402
    Phi4MultimodalQuantContext,
    _FP8Weight,
    _NVFP4Weight,
    calibrate_phi4_multimodal,
    is_quantized_checkpoint,
)

# Wide enough that packed-weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS, KV_HEADS = 512, 2048, 32, 2, 8, 4
HEAD_DIM = HIDDEN // HEADS
Q_DIM, KV_DIM = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

# HF fused projection stem -> (out_features, in_features)
_PROJECTIONS = {
    "self_attn.qkv_proj": (Q_DIM + 2 * KV_DIM, HIDDEN),
    "self_attn.o_proj": (HIDDEN, Q_DIM),
    "mlp.gate_up_proj": (2 * MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}
_EXCLUDE = ("model.layers.*.*.lora_A*", "model.layers.*.*.lora_B*", "lm_head")


def _config_json() -> dict:
    return {
        "model_type": "phi4mm",
        "architectures": ["Phi4MMForCausalLM"],
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
    }


def _write_checkpoint(
    path: Path,
    quantization: str,
    input_scale: float = 0.02,
    exclude: tuple[str, ...] = _EXCLUDE,
    skip_scales_for: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Write a tiny checkpoint: ModelOpt-style for fp8/nvfp4, plain bf16 for "none".

    Projections named in ``skip_scales_for`` are stored in bf16 without scales. Every
    projection also carries LoRA adapters, as the real checkpoints do.
    """
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }
    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in _PROJECTIONS.items():
            name = f"{base}.{stem}.base_layer"
            tensors[f"{base}.{stem}.lora_A.vision.weight"] = torch.zeros(4, in_f).bfloat16()
            tensors[f"{base}.{stem}.lora_B.vision.weight"] = torch.zeros(out_f, 4).bfloat16()
            if quantization == "none" or name in skip_scales_for:
                tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
                continue
            tensors[f"{name}.input_scale"] = torch.tensor(input_scale)
            if quantization == "fp8":
                weight = torch.randn(out_f, in_f, generator=gen).to(torch.float8_e4m3fn)
                tensors[f"{name}.weight"] = weight
                tensors[f"{name}.weight_scale"] = torch.tensor(0.01)
            else:
                tensors[f"{name}.weight"] = torch.randint(
                    0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
                )
                scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
                tensors[f"{name}.weight_scale"] = scale
                tensors[f"{name}.weight_scale_2"] = torch.tensor(0.003)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    if quantization != "none":
        quant = {"quant_algo": quantization.upper(), "exclude_modules": list(exclude)}
        (path / "hf_quant_config.json").write_text(json.dumps({"quantization": quant}))
    return tensors


def _calibrate(path: Path, quantization: str) -> Phi4MultimodalQuantContext:
    return calibrate_phi4_multimodal(path, ModelConfig.from_dir(path), graph_ops, quantization)


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_calibrate_splits_fused_projections_bit_exact(tmp_path: Path, quantization: str) -> None:
    """The fused qkv and gate_up weights split by rows into q/k/v and gate/up, bytes unchanged."""
    tensors = _write_checkpoint(tmp_path, quantization)
    ctx = _calibrate(tmp_path, quantization)

    # q, k, v, o, gate, up, down per layer
    assert len(ctx.scales) == LAYERS * 7
    for stem, rows in (
        ("w_q", (0, Q_DIM)),
        ("w_k", (Q_DIM, Q_DIM + KV_DIM)),
        ("w_v", (Q_DIM + KV_DIM, Q_DIM + 2 * KV_DIM)),
    ):
        entry = ctx.scales[f"layer.1.{stem}"]
        assert (entry.weight.out_features, entry.weight.in_features) == (rows[1] - rows[0], HIDDEN)
        assert entry.input_scale == pytest.approx(0.02)
        fused = tensors["model.layers.1.self_attn.qkv_proj.base_layer.weight"]
        source = fused.view(torch.uint8).numpy()[rows[0] : rows[1]]
        assert np.array_equal(entry.weight.packed, source)  # bit-exact
        assert entry.weight.packed.flags["C_CONTIGUOUS"]

    gate, up = ctx.scales["layer.0.w_gate"], ctx.scales["layer.0.w_up"]
    fused = tensors["model.layers.0.mlp.gate_up_proj.base_layer.weight"].view(torch.uint8).numpy()
    assert np.array_equal(gate.weight.packed, fused[:MLP])
    assert np.array_equal(up.weight.packed, fused[MLP:])

    down = ctx.scales["layer.1.w_down"]
    source = tensors["model.layers.1.mlp.down_proj.base_layer.weight"]
    assert np.array_equal(down.weight.packed, source.view(torch.uint8).numpy())
    if quantization == "fp8":
        assert isinstance(down.weight, _FP8Weight)
        assert down.weight.weight_scale == pytest.approx(0.01)
        assert ctx.disable_dual_gemm_fusion
    else:
        assert isinstance(down.weight, _NVFP4Weight)
        assert down.weight.global_scale == pytest.approx(0.003)
        scale_source = tensors["model.layers.0.mlp.gate_up_proj.base_layer.weight_scale"]
        assert np.array_equal(
            gate.weight.block_scale_f8, scale_source.view(torch.uint8).numpy()[:MLP]
        )
        assert gate.weight.block_scale_f8.shape == (MLP, HIDDEN // 16)
        assert not ctx.disable_dual_gemm_fusion


def test_calibrate_rejects_format_mismatch(tmp_path: Path) -> None:
    """An FP8 checkpoint is refused when nvfp4 is requested."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(ValueError, match="not a nvfp4 ModelOpt tensor"):
        _calibrate(tmp_path, "nvfp4")


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path, quantization: str) -> None:
    """A plain full-precision checkpoint cannot pass as quantized."""
    _write_checkpoint(tmp_path, "none")
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(tmp_path, quantization)


def test_calibrate_allows_excluded_projection(tmp_path: Path) -> None:
    """A projection ModelOpt left unquantized is accepted only when exclude_modules lists it."""
    name = "model.layers.0.mlp.down_proj.base_layer"
    _write_checkpoint(tmp_path, "fp8", skip_scales_for=(name,), exclude=(name, "lm_head"))
    ctx = _calibrate(tmp_path, "fp8")
    assert "layer.0.w_down" not in ctx.scales
    assert "layer.1.w_down" in ctx.scales

    # A wildcard entry covers it too.
    pattern = tmp_path / "pattern"
    _write_checkpoint(
        pattern, "fp8", skip_scales_for=(name,), exclude=("model.layers.0.mlp.down_proj.*",)
    )
    assert "layer.0.w_down" not in _calibrate(pattern, "fp8").scales

    other = tmp_path / "unlisted"
    _write_checkpoint(other, "fp8", skip_scales_for=(name,))
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(other, "fp8")


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(
    tmp_path: Path, quantization: str, bad_scale: float
) -> None:
    """A present but invalid input_scale gets its own error."""
    _write_checkpoint(tmp_path, quantization, input_scale=bad_scale)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        _calibrate(tmp_path, quantization)


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Formats other than fp8 and nvfp4 are not supported."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError):
        _calibrate(tmp_path, "int4")


def test_is_quantized_checkpoint(tmp_path: Path) -> None:
    """Quantized checkpoints are recognised so a build without the flag can refuse them."""
    _write_checkpoint(tmp_path / "plain", "none")
    _write_checkpoint(tmp_path / "fp8", "fp8")
    assert not is_quantized_checkpoint(tmp_path / "plain")
    assert is_quantized_checkpoint(tmp_path / "fp8")


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_load_weights_uses_shape_only_placeholders(tmp_path: Path, quantization: str) -> None:
    """Quantized projections are not loaded in full precision; the rest still are."""
    _write_checkpoint(tmp_path, quantization)
    config = ModelConfig.from_dir(tmp_path)
    ctx = _calibrate(tmp_path, quantization)
    weights = model._Phi4MultimodalModel().load_weights(str(tmp_path), config, quant_ctx=ctx)

    assert weights["layer.0.w_q"].shape == (HIDDEN, Q_DIM)
    assert weights["layer.0.w_k"].shape == (HIDDEN, KV_DIM)
    assert weights["layer.0.w_gate"].shape == (HIDDEN, MLP)
    assert weights["layer.0.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_gate"].strides == (0, 0)
    assert weights["_mlp_size"] == MLP
    assert weights["_attention_size"] == Q_DIM
    # The embedding, norms and lm_head are not quantized and are still loaded.
    assert weights["embedding"].shape == (VOCAB, HIDDEN)
    assert weights["w_out"].shape == (HIDDEN, VOCAB)


def _request(path: Path, **overrides) -> BuildRequest:
    values = {
        "model_dir": path,
        "output_path": path / "out.bundle",
        "family": "phi4_multimodal",
        "task": "vision_language_generation",
        "precision": "fp16",
        "max_sequence_length": 4,
    }
    values.update(overrides)
    return BuildRequest(**values)


def _native_build(request: BuildRequest) -> None:
    """Run the family's native build only (the complete-network route is not under test)."""
    captured = {}

    def fake_dispatch(request, writer, native):
        captured["native"] = native

    import families.phi4_multimodal.edge_llm.dispatch as dispatch

    original = dispatch.build
    dispatch.build = fake_dispatch
    try:
        model.build(request, Mock())
        captured["native"](request, Mock())
    finally:
        dispatch.build = original


def test_build_refuses_quantized_checkpoint_without_flag(tmp_path: Path) -> None:
    """A quantized checkpoint built without --quantization is refused, not read as weights."""
    _write_checkpoint(tmp_path, "nvfp4")
    with pytest.raises(ValueError, match="--quantization"):
        _native_build(_request(tmp_path))


def test_build_rejects_unsupported_precision(tmp_path: Path) -> None:
    """Quantized builds need fp16."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError, match="require fp16"):
        _native_build(_request(tmp_path, precision="fp32", quantization="fp8"))


def test_build_rejects_quantization_with_tensor_parallelism(tmp_path: Path) -> None:
    """Tensor-parallel builds are not supported, quantized or not."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError, match="single-device"):
        _native_build(_request(tmp_path, quantization="fp8", tensor_parallel_size=2))


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Formats other than fp8 and nvfp4 are refused."""
    _write_checkpoint(tmp_path, "none")
    with pytest.raises(NotImplementedError, match="quantization"):
        _native_build(_request(tmp_path, quantization="int4"))


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_engine_builds_with_packed_weights(tmp_path: Path, quantization: str) -> None:
    """The quantized decoder engine builds and is smaller than the unquantized one.

    Both engines are built from the production weight loader, so the quantized
    one sees the same shape-only placeholders as a real build.
    """
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, "none")
    _write_checkpoint(quant_dir, quantization)
    config = ModelConfig.from_dir(plain_dir)
    config.raw["_decoder_engine_role"] = "decode"
    ctx = _calibrate(quant_dir, quantization)
    family = model._Phi4MultimodalModel()

    plain = family.build_engine(
        config, family.load_weights(str(plain_dir), config), 4, precision="fp16"
    )
    quantized = family.build_engine(
        config,
        family.load_weights(str(quant_dir), config, quant_ctx=ctx),
        4,
        precision="fp16",
        quant_ctx=ctx,
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves and FP4 quarters the fp16 projection bytes,
    # so a plan that stored 16-bit weights again would miss these bounds.
    bound = {"fp8": 0.6, "nvfp4": 0.4}[quantization]
    assert len(quantized) < bound * len(plain), (len(quantized), len(plain))
