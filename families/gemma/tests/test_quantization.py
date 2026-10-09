# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma FP8 self-quantization: weight quantizer, scale lookup, weight loading, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import graph_ops, quantization  # noqa: E402
from ..build_request import BuildRequest, coerce_request  # noqa: E402
from ..checkpoint_mapper import WeightDict, load_standard_weights  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    GemmaQuantContext,
    calibrate_gemma_fp8,
    load_activation_scales,
    quantize_fp8_weight,
    scale_key,
)
from ..standard_decoder_builder import build_standard_decoder_engine  # noqa: E402

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
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
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _config() -> ModelConfig:
    return ModelConfig.create_tiny(
        "gemma2",
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        intermediate_size=MLP,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HIDDEN // HEADS,
        hidden_activation="gelu_pytorch_tanh",
    )


def _write_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Write a tiny BF16 Gemma checkpoint."""
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen).bfloat16(),
        "model.norm.weight": torch.zeros(HIDDEN).bfloat16(),
    }
    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[f"{base}.{norm}.weight"] = torch.zeros(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in _PROJECTIONS.items():
            tensors[f"{base}.{stem}.weight"] = torch.randn(out_f, in_f, generator=gen).bfloat16()
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
    return tensors


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    return {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(LAYERS)
        for stem in _WEIGHT_STEMS.values()
        if stem not in skip
    }


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)

    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    weight = np.full((4, 4), 3.0, dtype=np.float32)
    packed, _ = quantize_fp8_weight(weight)
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    with pytest.raises(ValueError):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_scale_key_identifies_model() -> None:
    assert scale_key(_config()) == f"gemma2-h{HIDDEN}-l{LAYERS}-i{MLP}"


def test_missing_scale_entry_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config())


def test_committed_scales_cover_every_projection() -> None:
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    for key, scales in json.loads(path.read_text(encoding="utf-8")).items():
        assert scales, key
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        assert len(scales) == len(layers) * len(_WEIGHT_STEMS), key


def test_calibrate_quantizes_the_checkpoint_weights(tmp_path: Path, monkeypatch) -> None:
    tensors = _write_checkpoint(tmp_path)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_gemma_fp8(tmp_path, _config(), graph_ops, model_prefix="model")

    assert len(ctx.scales) == LAYERS * len(_PROJECTIONS)
    entry = ctx.scales["layer.1.w_down"]
    out_f, in_f = _PROJECTIONS["mlp.down_proj"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = tensors["model.layers.1.mlp.down_proj.weight"].float().numpy()
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)


def test_projection_without_a_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _activation_scales(("w_down",))
    )
    ctx = calibrate_gemma_fp8(tmp_path, _config(), graph_ops, model_prefix="model")
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_load_weights_uses_shape_only_placeholders(tmp_path: Path, monkeypatch) -> None:
    _write_checkpoint(tmp_path)
    config = _config()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _activation_scales(("w_down",))
    )
    ctx = calibrate_gemma_fp8(tmp_path, config, graph_ops, model_prefix="model")
    weights = load_standard_weights(tmp_path, config, precision="bf16", quant_ctx=ctx)

    assert weights["layer.0.w_q"].shape == (HIDDEN, HIDDEN)
    assert weights["layer.0.w_gate"].shape == (HIDDEN, MLP)
    # Placeholders must not hold a full-size copy of the weight.
    assert weights["layer.0.w_gate"].strides == (0, 0)
    # An unquantized projection is still loaded for real.
    assert weights["layer.0.w_down"].shape == (MLP, HIDDEN)
    assert weights["layer.0.w_down"].strides != (0, 0)
    assert weights["_mlp_size"] == MLP
    assert weights["_kv_attention_size"] == HIDDEN


@pytest.mark.parametrize("value", [None, "none", "fp8"])
def test_request_accepts_supported_quantization(value) -> None:
    legacy = SimpleNamespace(
        model_dir=Path("m"),
        output_path=Path("o"),
        family="gemma",
        task="text_generation",
        precision="bf16",
        backend="trt",
        max_sequence_length=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        fp32_layers=(),
        dynamic_kv_cache=False,
        verbose=False,
        graph_transform=None,
        quantization=value,
    )
    request = coerce_request(legacy)
    assert isinstance(request, BuildRequest)
    assert request.quantization == value


def _fake_weights() -> WeightDict:
    rng = np.random.RandomState(0)
    weights = WeightDict()
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float16)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm", "pre_ffn_norm", "post_ffn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        for stem, shape in (
            ("w_q", (HIDDEN, HIDDEN)),
            ("w_k", (HIDDEN, HIDDEN)),
            ("w_v", (HIDDEN, HIDDEN)),
            ("w_o", (HIDDEN, HIDDEN)),
            ("w_gate", (HIDDEN, MLP)),
            ("w_up", (HIDDEN, MLP)),
            ("w_down", (MLP, HIDDEN)),
        ):
            weights[f"{prefix}.{stem}"] = rng.randn(*shape).astype(np.float16)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_out"] = rng.randn(HIDDEN, VOCAB).astype(np.float16)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = HIDDEN
    weights["_mlp_size"] = MLP
    return weights


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tmp_path: Path, monkeypatch) -> None:
    """The quantized engine builds and stores FP8 (not full-precision) weights."""
    _write_checkpoint(tmp_path)
    config = _config()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_gemma_fp8(tmp_path, config, graph_ops, model_prefix="model")
    assert isinstance(ctx, GemmaQuantContext)

    kwargs = {"precision": "bf16", "activation": "gelu_pytorch_tanh"}
    plain = build_standard_decoder_engine(config, _fake_weights(), 4, **kwargs)
    quantized = build_standard_decoder_engine(
        config, _fake_weights(), 4, quant_ctx=ctx, **kwargs
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves the bf16 projection bytes.
    assert len(quantized) < len(plain)
