# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H FP8 / NVFP4 quantization: checkpoint reading, placeholders, refusals, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from families.nemotron_h import graph_ops, model  # noqa: E402
from families.nemotron_h.build_request import BuildRequest, coerce_request  # noqa: E402
from families.nemotron_h.config import ModelConfig  # noqa: E402
from families.nemotron_h.quantization import (  # noqa: E402
    NemotronHQuantContext,
    _FP8Weight,
    _NVFP4Weight,
    calibrate_nemotron_h,
    is_quantized_checkpoint,
)

# Every quantized input dimension is a multiple of the 16-element NVFP4 block.
VOCAB, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP = 32, 256, 4, 2, 64, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
MAMBA_HEADS, MAMBA_HEAD_DIM, GROUPS, D_STATE, D_CONV = 4, 64, 2, 16, 4
D_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM
CONV_DIM = D_INNER + 2 * GROUPS * D_STATE
PROJ_DIM = D_INNER + CONV_DIM + MAMBA_HEADS
PATTERN = "M-*M"
LAYER_TYPES = ["mamba2", "mlp", "attention", "mamba2"]
# ModelOpt leaves the mixer of the first layer unquantized (like the real NVFP4 export).
UNQUANTIZED_LAYERS = (0,)

_MAMBA = {"in_proj": (PROJ_DIM, HIDDEN), "out_proj": (HIDDEN, D_INNER)}
_MLP = {"up_proj": (MLP, HIDDEN), "down_proj": (HIDDEN, MLP)}
_ATTENTION = {
    "q_proj": (ATTENTION, HIDDEN),
    "k_proj": (KV, HIDDEN),
    "v_proj": (KV, HIDDEN),
    "o_proj": (HIDDEN, ATTENTION),
}


def _config_json() -> dict:
    return {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "layer_norm_epsilon": 1e-5,
        "max_position_embeddings": 64,
        "hybrid_override_pattern": PATTERN,
        "mamba_num_heads": MAMBA_HEADS,
        "mamba_head_dim": MAMBA_HEAD_DIM,
        "n_groups": GROUPS,
        "ssm_state_size": D_STATE,
        "conv_kernel": D_CONV,
    }


def _write_checkpoint(
    path: Path,
    *,
    fmt: str | None,
    input_scale: float = 0.02,
    list_excluded: bool = True,
) -> dict[str, torch.Tensor]:
    """Write a tiny hybrid checkpoint. ``fmt`` is ``"fp8"``, ``"nvfp4"`` or None (BF16)."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "backbone.norm_f.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
    }
    excluded: list[str] = ["lm_head"]

    def add_projection(name: str, out_f: int, in_f: int, quantize: bool) -> None:
        if not quantize or fmt is None:
            tensors[f"{name}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
            if fmt is not None:
                excluded.append(name)
            return
        if fmt == "nvfp4":
            tensors[f"{name}.weight"] = torch.randint(
                0, 256, (out_f, in_f // 2), generator=gen, dtype=torch.uint8
            )
            scale = (torch.rand(out_f, in_f // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn)
            tensors[f"{name}.weight_scale"] = scale
            tensors[f"{name}.weight_scale_2"] = torch.tensor(0.004)
        else:
            weight = torch.randn(out_f, in_f, generator=gen).to(torch.float8_e4m3fn)
            tensors[f"{name}.weight"] = weight
            tensors[f"{name}.weight_scale"] = torch.tensor(0.01)
        tensors[f"{name}.input_scale"] = torch.tensor(input_scale)

    for layer, kind in enumerate(LAYER_TYPES):
        base = f"backbone.layers.{layer}"
        tensors[f"{base}.norm.weight"] = torch.ones(HIDDEN).bfloat16()
        quantize = layer not in UNQUANTIZED_LAYERS
        if kind == "mamba2":
            tensors[f"{base}.mixer.conv1d.weight"] = torch.randn(
                CONV_DIM, 1, D_CONV, generator=gen
            ).bfloat16()
            tensors[f"{base}.mixer.conv1d.bias"] = torch.zeros(CONV_DIM).bfloat16()
            tensors[f"{base}.mixer.A_log"] = torch.zeros(MAMBA_HEADS)
            tensors[f"{base}.mixer.D"] = torch.ones(MAMBA_HEADS)
            tensors[f"{base}.mixer.dt_bias"] = torch.ones(MAMBA_HEADS)
            tensors[f"{base}.mixer.norm.weight"] = torch.ones(D_INNER).bfloat16()
            projections = _MAMBA
        elif kind == "mlp":
            projections = _MLP
        else:
            projections = _ATTENTION
            quantize = False
        for stem, (out_f, in_f) in projections.items():
            add_projection(f"{base}.mixer.{stem}", out_f, in_f, quantize)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    if fmt is not None:
        quant = {"quant_algo": fmt.upper(), "group_size": 16}
        if list_excluded:
            quant["exclude_modules"] = excluded
        (path / "hf_quant_config.json").write_text(json.dumps({"quantization": quant}))
    return tensors


def _calibrate(path: Path, fmt: str) -> NemotronHQuantContext:
    return calibrate_nemotron_h(path, graph_ops, fmt, layer_types=LAYER_TYPES)


@pytest.mark.parametrize(("fmt", "weight_type"), [("nvfp4", _NVFP4Weight), ("fp8", _FP8Weight)])
def test_calibrate_reads_quantized_projections_bit_exact(
    tmp_path: Path, fmt: str, weight_type: type
) -> None:
    """Mamba and MLP projections are read as the checkpoint's own bytes; the rest are skipped."""
    tensors = _write_checkpoint(tmp_path, fmt=fmt)
    ctx = _calibrate(tmp_path, fmt)

    # layer 0 mixer is left unquantized and attention is never quantized
    assert set(ctx.scales) == {
        "layer.1.w_up",
        "layer.1.w_down",
        "layer.3.mamba_in_proj",
        "layer.3.mamba_out_proj",
    }
    entry = ctx.scales["layer.3.mamba_in_proj"]
    assert isinstance(entry.weight, weight_type)
    assert (entry.weight.out_features, entry.weight.in_features) == (PROJ_DIM, HIDDEN)
    assert np.array_equal(
        entry.weight.packed,
        tensors["backbone.layers.3.mixer.in_proj.weight"].view(torch.uint8).numpy(),
    )
    assert entry.input_scale == pytest.approx(0.02)
    assert ctx.should_quantize("layer.1.w_down")
    assert not ctx.should_quantize("layer.0.mamba_in_proj")
    assert not ctx.should_quantize("layer.2.w_q")


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8"])
def test_calibrate_rejects_unquantized_checkpoint(tmp_path: Path, fmt: str) -> None:
    """A BF16 checkpoint fails with a format error instead of silently staying unquantized."""
    _write_checkpoint(tmp_path, fmt=None)
    with pytest.raises(ValueError, match="has no .* scales"):
        _calibrate(tmp_path, fmt)


@pytest.mark.parametrize(("checkpoint", "requested"), [("nvfp4", "fp8"), ("fp8", "nvfp4")])
def test_calibrate_rejects_the_other_format(
    tmp_path: Path, checkpoint: str, requested: str
) -> None:
    """An FP8 checkpoint cannot be built as NVFP4, and the other way round."""
    _write_checkpoint(tmp_path, fmt=checkpoint)
    with pytest.raises(ValueError, match=f"is not a {requested} ModelOpt tensor"):
        _calibrate(tmp_path, requested)


def test_calibrate_requires_unquantized_projections_to_be_excluded(tmp_path: Path) -> None:
    """A projection without scales is an error unless the checkpoint lists it as excluded."""
    _write_checkpoint(tmp_path, fmt="nvfp4", list_excluded=False)
    with pytest.raises(ValueError, match="not listed in exclude_modules"):
        _calibrate(tmp_path, "nvfp4")


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    """Only fp8 and nvfp4 are supported."""
    with pytest.raises(NotImplementedError):
        calibrate_nemotron_h(tmp_path, graph_ops, "int8", layer_types=LAYER_TYPES)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_calibrate_rejects_invalid_input_scale(tmp_path: Path, fmt: str, bad: float) -> None:
    """A zero, negative or NaN input_scale is reported as such."""
    _write_checkpoint(tmp_path, fmt=fmt, input_scale=bad)
    with pytest.raises(ValueError, match="input_scale must be one positive finite value"):
        _calibrate(tmp_path, fmt)


def test_is_quantized_checkpoint(tmp_path: Path) -> None:
    """Quantized checkpoints are recognised so a build without the flag can refuse them."""
    _write_checkpoint(tmp_path / "nvfp4", fmt="nvfp4")
    _write_checkpoint(tmp_path / "fp8", fmt="fp8")
    _write_checkpoint(tmp_path / "plain", fmt=None)
    assert is_quantized_checkpoint(tmp_path / "nvfp4")
    assert is_quantized_checkpoint(tmp_path / "fp8")
    assert not is_quantized_checkpoint(tmp_path / "plain")
    assert not is_quantized_checkpoint(tmp_path / "missing")


def test_build_request_accepts_quantization(tmp_path: Path) -> None:
    """quantization is a real request field; 'none' and None both mean unquantized."""

    def make(**extra) -> BuildRequest:
        return BuildRequest(
            model_dir=tmp_path,
            output_path=tmp_path / "out",
            family="nemotron_h",
            task="text_generation",
            precision="fp16",
            **extra,
        )

    assert coerce_request(make(quantization="fp8")).quantization == "fp8"
    assert make().quantization is None


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8"])
def test_load_weights_uses_shape_only_placeholders(tmp_path: Path, fmt: str) -> None:
    """Quantized projections get shape-only placeholders; unquantized ones are still loaded."""
    _write_checkpoint(tmp_path, fmt=fmt)
    config = ModelConfig.from_dir(tmp_path)
    ctx = _calibrate(tmp_path, fmt)
    weights = model._NemotronHModel().load_weights(str(tmp_path), config, ctx)

    assert weights["layer.3.mamba_in_proj"].shape == (HIDDEN, PROJ_DIM)
    assert weights["layer.3.mamba_out_proj"].shape == (D_INNER, HIDDEN)
    assert weights["layer.1.w_up"].shape == (HIDDEN, MLP)
    assert weights["layer.1.w_down"].shape == (MLP, HIDDEN)
    # Placeholders must not hold a full-size copy of the weight.
    for name in ("layer.3.mamba_in_proj", "layer.3.mamba_out_proj", "layer.1.w_up"):
        assert weights[name].strides == (0, 0)
    # Unquantized projections are real arrays.
    assert weights["layer.0.mamba_in_proj"].strides != (0, 0)
    assert weights["layer.2.w_q"].strides != (0, 0)
    assert weights["layer.2.w_q"].shape == (HIDDEN, ATTENTION)
    assert weights["w_lm_head"].shape == (HIDDEN, VOCAB)


def _request(path: Path, **overrides) -> BuildRequest:
    values = {
        "model_dir": path,
        "output_path": path / "out.bundle",
        "family": "nemotron_h",
        "task": "text_generation",
        "precision": "fp16",
        "max_sequence_length": 4,
    }
    values.update(overrides)
    return BuildRequest(**values)


def test_build_refuses_quantized_checkpoint_without_flag(tmp_path: Path) -> None:
    """A quantized checkpoint built without --quantization is refused, not read as weights."""
    _write_checkpoint(tmp_path, fmt="nvfp4")
    with pytest.raises(ValueError, match="--quantization"):
        model.build(_request(tmp_path, precision="fp32"), Mock())


def test_build_requires_fp16_for_quantization(tmp_path: Path) -> None:
    """Quantized builds need fp16."""
    _write_checkpoint(tmp_path, fmt="fp8")
    with pytest.raises(ValueError, match="require precision fp16"):
        model.build(_request(tmp_path, precision="fp32", quantization="fp8"), Mock())


def test_build_rejects_quantization_with_tensor_parallelism(tmp_path: Path) -> None:
    """Tensor-parallel quantized builds are not supported."""
    _write_checkpoint(tmp_path, fmt="fp8")
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        model.build(_request(tmp_path, quantization="fp8", tensor_parallel_size=2), Mock())


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Formats other than fp8 and nvfp4 are refused."""
    _write_checkpoint(tmp_path, fmt=None)
    with pytest.raises(NotImplementedError, match="quantization"):
        model.build(_request(tmp_path, precision="fp32", quantization="int8"), Mock())


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("fmt", ["nvfp4", "fp8"])
def test_engine_builds_with_packed_weights(tmp_path: Path, fmt: str) -> None:
    """The quantized engine builds and stores packed (not full-precision) weights."""
    plain_dir, quant_dir = tmp_path / "plain", tmp_path / "quant"
    _write_checkpoint(plain_dir, fmt=None)
    _write_checkpoint(quant_dir, fmt=fmt)
    instance = model._NemotronHModel()

    config = ModelConfig.from_dir(plain_dir)
    plain = instance.build_engine(
        config, instance.load_weights(str(plain_dir), config), 4, precision="fp16"
    )

    config = ModelConfig.from_dir(quant_dir)
    ctx = _calibrate(quant_dir, fmt)
    weights = instance.load_weights(str(quant_dir), config, ctx)
    quantized = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits"} <= names
    # The projections are 8 or 4 bits instead of 16; the embedding and lm_head are not.
    assert len(quantized) < len(plain)
