# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL FP8 / NVFP4 quantization: checkpoint reading, weight loading, refusals, engine build."""

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

from .. import graph_ops, model  # noqa: E402
from ..checkpoint_mapper import load_standard_weights  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    QwenVLQuantContext,
    _FP8Weight,
    _NVFP4Weight,
    calibrate_qwen_vl,
    is_quantized_checkpoint,
)

# Wide enough that packed-weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS = 512, 2048, 32, 2, 4

_PROJECTIONS = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (HIDDEN, HIDDEN),
    "self_attn.v_proj": (HIDDEN, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (MLP, HIDDEN),
    "mlp.up_proj": (MLP, HIDDEN),
    "mlp.down_proj": (HIDDEN, MLP),
}


def _config_json() -> dict:
    return {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "vision_config": {},
    }


def _write_checkpoint(
    path: Path,
    quantization: str,
    input_scale: float = 0.02,
    exclude: tuple[str, ...] = ("visual*", "lm_head"),
    skip_scales_for: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Write a tiny checkpoint: ModelOpt-style for fp8/nvfp4, plain bf16 for "none".

    Projections named in ``skip_scales_for`` are stored in bf16 without scales.
    """
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "visual.patch_embed.proj.weight": torch.randn(8, 3, generator=gen).bfloat16(),
    }
    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        for stem in ("q", "k", "v"):
            tensors[f"{base}.self_attn.{stem}_proj.bias"] = torch.zeros(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in _PROJECTIONS.items():
            name = f"{base}.{stem}"
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


def _calibrate(path: Path, quantization: str) -> QwenVLQuantContext:
    return calibrate_qwen_vl(path, ModelConfig.from_dir(path), graph_ops, quantization)


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_calibrate_reads_checkpoint_bytes(tmp_path: Path, quantization: str) -> None:
    """Weights and scales come from the checkpoint bytes, with no re-quantization."""
    tensors = _write_checkpoint(tmp_path, quantization)
    ctx = _calibrate(tmp_path, quantization)

    assert len(ctx.scales) == LAYERS * len(_PROJECTIONS)
    entry = ctx.scales["layer.1.w_down"]
    out_f, in_f = _PROJECTIONS["mlp.down_proj"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = tensors["model.layers.1.mlp.down_proj.weight"]
    assert np.array_equal(entry.weight.packed, source.view(torch.uint8).numpy())  # bit-exact
    if quantization == "fp8":
        assert isinstance(entry.weight, _FP8Weight)
        assert entry.weight.weight_scale == pytest.approx(0.01)
        assert ctx.disable_dual_gemm_fusion
    else:
        assert isinstance(entry.weight, _NVFP4Weight)
        assert entry.weight.global_scale == pytest.approx(0.003)
        assert entry.weight.block_scale_f8.shape == (out_f, in_f // 16)
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
    name = "model.layers.0.mlp.down_proj"
    _write_checkpoint(tmp_path, "fp8", skip_scales_for=(name,), exclude=(name, "lm_head"))
    ctx = _calibrate(tmp_path, "fp8")
    assert "layer.0.w_down" not in ctx.scales
    assert "layer.1.w_down" in ctx.scales

    other = tmp_path / "unlisted"
    _write_checkpoint(other, "fp8", skip_scales_for=(name,))
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(other, "fp8")


def test_calibrate_matches_exclude_modules_as_glob_patterns(tmp_path: Path) -> None:
    """ModelOpt writes exclude_modules as glob patterns, not only exact module names."""
    name = "model.layers.0.mlp.down_proj"
    _write_checkpoint(
        tmp_path, "fp8", skip_scales_for=(name,), exclude=("model.layers.*.mlp.down_proj", "lm_head")
    )
    ctx = _calibrate(tmp_path, "fp8")
    assert "layer.0.w_down" not in ctx.scales


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
    weights = load_standard_weights(tmp_path, config, precision="fp16", quant_ctx=ctx)

    assert weights["layer.0.w_q"].shape == (HIDDEN, HIDDEN)
    assert weights["layer.0.w_gate"].shape == (HIDDEN, MLP)
    assert weights["layer.0.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_gate"].strides == (0, 0)
    assert weights["_mlp_size"] == MLP
    assert weights["_kv_attention_size"] == HIDDEN
    # The biases and lm_head are not quantized and are still loaded.
    assert weights["layer.0.q_bias"].shape == (HIDDEN,)
    assert weights["w_out"].shape == (HIDDEN, VOCAB)


def _request(path: Path, **overrides) -> BuildRequest:
    values = {
        "model_dir": path,
        "output_path": path / "out.bundle",
        "family": "qwen_vl",
        "task": "vision_language_generation",
        "precision": "fp16",
        "max_sequence_length": 4,
    }
    values.update(overrides)
    return BuildRequest(**values)


def test_build_refuses_quantized_checkpoint_without_flag(tmp_path: Path) -> None:
    """A quantized checkpoint built without --quantization is refused, not read as weights."""
    _write_checkpoint(tmp_path, "nvfp4")
    with pytest.raises(ValueError, match="--quantization"):
        model.build(_request(tmp_path, precision="fp32"), Mock())


def test_build_rejects_unsupported_precision(tmp_path: Path) -> None:
    """Quantized builds need fp16."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError, match="require fp16"):
        model.build(_request(tmp_path, precision="fp32", quantization="fp8"), Mock())


def test_build_rejects_quantization_with_tensor_parallelism(tmp_path: Path) -> None:
    """Tensor-parallel quantized builds are not supported."""
    _write_checkpoint(tmp_path, "fp8")
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        model.build(_request(tmp_path, quantization="fp8", tensor_parallel_size=2), Mock())


def test_build_rejects_quantization_for_qwen3_vl(tmp_path: Path) -> None:
    """Only Qwen2.5-VL has a ModelOpt checkpoint; Qwen3-VL (deepstack) is refused."""
    _write_checkpoint(tmp_path, "fp8")
    config = _config_json()
    config["model_type"] = "qwen3_vl"
    config["vision_config"] = {"deepstack_visual_indexes": [5, 11, 17]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="Qwen2.5-VL"):
        model.build(_request(tmp_path, quantization="fp8"), Mock())


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Formats other than fp8 and nvfp4 are refused."""
    _write_checkpoint(tmp_path, "none")
    with pytest.raises(NotImplementedError, match="quantization"):
        model.build(_request(tmp_path, quantization="int4"), Mock())


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
    config.raw["_active_split_decoder_build"] = True
    config.raw["_decoder_engine_role"] = "decode"
    # Same decode attention backend as quantized builds: native IAttention in the fp16
    # decode engine does not compile on TensorRT 11.4.
    config.raw["_family_build_options"] = {"qwen_vl_decoder": {"decode_attention": "decomposed", "prefill_attention": "decomposed"}}
    ctx = _calibrate(quant_dir, quantization)

    plain = build_standard_decoder_engine(
        config,
        load_standard_weights(plain_dir, config, precision="fp16"),
        4,
        precision="fp16",
        embed_input=True,
    )
    quantized = build_standard_decoder_engine(
        config,
        load_standard_weights(quant_dir, config, precision="fp16", quant_ctx=ctx),
        4,
        precision="fp16",
        quant_ctx=ctx,
        embed_input=True,
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves and FP4 quarters the fp16 projection bytes,
    # so a plan that stored 16-bit weights again would miss these bounds.
    bound = {"fp8": 0.6, "nvfp4": 0.4}[quantization]
    assert len(quantized) < bound * len(plain), (len(quantized), len(plain))


def test_build_rejects_quantization_for_other_model_types(tmp_path: Path) -> None:
    """Only the qwen2_5_vl model type is accepted, even without Qwen3-VL's deepstack config."""
    _write_checkpoint(tmp_path, "fp8")
    config = _config_json()
    config["model_type"] = "qwen2_vl"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="Qwen2.5-VL"):
        model.build(_request(tmp_path, quantization="fp8"), Mock())
