# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    WhisperQuantContext,
    _projections,
    calibrate_whisper,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, MEL, SRC, TGT, VOCAB, LAYERS, HEADS = 128, 256, 16, 32, 32, 100, 2, 4


def _write_checkpoint(path: Path) -> ModelConfig:
    """Write a random tiny Whisper checkpoint (HF tensor names) and return its config."""
    rng = np.random.default_rng(0)

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    t: dict[str, np.ndarray] = {
        "model.encoder.conv1.weight": w(D, MEL, 3),
        "model.encoder.conv1.bias": w(D),
        "model.encoder.conv2.weight": w(D, D, 3),
        "model.encoder.conv2.bias": w(D),
        "model.encoder.embed_positions.weight": w(SRC, D),
        "model.encoder.layer_norm.weight": np.ones(D, np.float32),
        "model.encoder.layer_norm.bias": np.zeros(D, np.float32),
        "model.decoder.embed_tokens.weight": w(VOCAB, D),
        "model.decoder.embed_positions.weight": w(TGT, D),
        "model.decoder.layer_norm.weight": np.ones(D, np.float32),
        "model.decoder.layer_norm.bias": np.zeros(D, np.float32),
    }

    def attention(prefix: str) -> None:
        for proj in ("q", "k", "v", "out"):
            t[f"{prefix}.{proj}_proj.weight"] = w(D, D)
            if proj != "k":
                t[f"{prefix}.{proj}_proj.bias"] = w(D)

    def norm(prefix: str) -> None:
        t[f"{prefix}.weight"] = np.ones(D, np.float32)
        t[f"{prefix}.bias"] = np.zeros(D, np.float32)

    def mlp(prefix: str) -> None:
        t[f"{prefix}.fc1.weight"], t[f"{prefix}.fc1.bias"] = w(FFN, D), w(FFN)
        t[f"{prefix}.fc2.weight"], t[f"{prefix}.fc2.bias"] = w(D, FFN), w(D)

    for i in range(LAYERS):
        enc = f"model.encoder.layers.{i}"
        attention(f"{enc}.self_attn")
        norm(f"{enc}.self_attn_layer_norm")
        norm(f"{enc}.final_layer_norm")
        mlp(enc)
        dec = f"model.decoder.layers.{i}"
        attention(f"{dec}.self_attn")
        attention(f"{dec}.encoder_attn")
        norm(f"{dec}.self_attn_layer_norm")
        norm(f"{dec}.encoder_attn_layer_norm")
        norm(f"{dec}.final_layer_norm")
        mlp(dec)

    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
                "d_model": D,
                "vocab_size": VOCAB,
                "encoder_layers": LAYERS,
                "decoder_layers": LAYERS,
                "encoder_attention_heads": HEADS,
                "decoder_attention_heads": HEADS,
                "encoder_ffn_dim": FFN,
                "decoder_ffn_dim": FFN,
                "num_mel_bins": MEL,
                "max_source_positions": SRC,
                "max_target_positions": TGT,
                "activation_function": "gelu",
                "eos_token_id": 3,
            }
        ),
        encoding="utf-8",
    )
    return ModelConfig.from_dir(path)


def _all_scales(config: ModelConfig, value: float = 0.02) -> dict[str, float]:
    return {name: value for _, name in _projections(config)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny checkpoint plus an activation-scale table that covers every projection."""
    config = _write_checkpoint(tmp_path)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: _all_scales(cfg))
    return tmp_path, config


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the original weight within e4m3 precision."""
    weight = np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= 0.07 * float(np.abs(weight).max())


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_every_projection_is_quantized(tiny) -> None:
    """Encoder, decoder self-attention, cross-attention and MLP projections all get FP8 weights."""
    path, config = tiny
    ctx = calibrate_whisper(path, graph_ops, "fp8", config)
    # encoder: q k v fc1 fc2; decoder: q k v + cross q k v o + fc1 fc2
    assert len(ctx.scales) == LAYERS * 5 + LAYERS * 9
    # self-attention output projections stay fp16 (TensorRT 11.4 cannot compile them in FP8)
    assert not ctx.should_quantize("layer.0.w_o")
    assert not ctx.should_quantize("enc_layer.0.w_o")
    assert ctx.should_quantize("enc_layer.0.w_q")
    assert ctx.should_quantize("layer.1.cross_w_k")
    assert ctx.should_quantize("layer.1.w_fc2")
    entry = ctx.scales["layer.0.cross_w_v"]
    assert (entry.weight.out_features, entry.weight.in_features) == (D, D)
    assert ctx.scales["enc_layer.0.w_fc1"].weight.packed.shape == (FFN, D)
    assert ctx.has_fp8_weights


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    config = _write_checkpoint(tmp_path)
    scales = _all_scales(config)
    del scales["layer.0.cross_w_k"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    ctx = calibrate_whisper(tmp_path, graph_ops, "fp8", config)
    assert not ctx.should_quantize("layer.0.cross_w_k")
    assert ctx.should_quantize("layer.0.cross_w_v")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["model.encoder.layers.0.fc1.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_whisper(path, graph_ops, "fp8", config)


def test_already_quantized_projection_without_scale_is_rejected(tiny, monkeypatch) -> None:
    """The check runs before projections without an activation scale are skipped."""
    path, config = tiny
    tensors = safetensors_numpy.load_file(str(path / "model.safetensors"))
    tensors["model.decoder.layers.0.self_attn.out_proj.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(path / "model.safetensors"))
    scales = _all_scales(config)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_whisper(path, graph_ops, "fp8", config)


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    path, config = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_whisper(path, graph_ops, "nvfp4", config)


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    config = _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_whisper(tmp_path, graph_ops, "fp8", config)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"whisper-h(\d+)-e(\d+)-d(\d+)-m(\d+)", key)
        assert match, key
        hidden, enc, dec, mel = (int(g) for g in match.groups())
        config = ModelConfig(
            model_type="whisper",
            hidden_size=hidden,
            raw={"encoder_layers": enc, "decoder_layers": dec, "num_mel_bins": mel},
        )
        assert scale_key(config) == key
        assert {name for _, name in _projections(config)} == set(scales), key
        assert all(value > 0 for value in scales.values()), key


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps HF module paths to internal weight names."""
    name = calibrate_fp8_scales.weight_name
    assert name("model.encoder.layers.3.self_attn.out_proj") == "enc_layer.3.w_o"
    assert name("model.encoder.layers.0.fc1") == "enc_layer.0.w_fc1"
    assert name("model.decoder.layers.2.self_attn.k_proj") == "layer.2.w_k"
    assert name("model.decoder.layers.2.encoder_attn.v_proj") == "layer.2.cross_w_v"
    assert name("model.decoder.layers.2.fc2") == "layer.2.w_fc2"
    assert name("model.decoder.layers.2.fc2.base") is None
    assert name("proj_out") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_clips", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, c, n: seen.update(calibrate=(d, c, n))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "clips")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--clips", "clips"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "clips", 96)
    assert seen["calibrate"] == (tmp_path, Path("clips"), 96)


def test_quantized_engines_build(tiny) -> None:
    """The encoder and decoder engines build with FP8 projections and are smaller than fp16."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    path, config = tiny
    whisper = model._WhisperModel()
    config.raw["_model_dir"] = str(path)
    config.raw["_fp32_layers"] = ()
    weights = whisper.load_weights(str(path), config)
    ctx = calibrate_whisper(path, graph_ops, "fp8", config)

    plain_decoder = whisper.build_engine(config, weights, 8, precision="fp16")
    plain_encoder = whisper.build_vision_engine(str(path), config, weights, precision="fp16")
    fp8_decoder = whisper.build_engine(config, weights, 8, precision="fp16", quant_ctx=ctx)
    fp8_encoder = whisper.build_vision_engine(
        str(path), config, weights, precision="fp16", quant_ctx=ctx
    )

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    decoder = runtime.deserialize_cuda_engine(fp8_decoder)
    names = {decoder.get_tensor_name(i) for i in range(decoder.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "cross_k_0"} <= names
    encoder = runtime.deserialize_cuda_engine(fp8_encoder)
    assert encoder.get_tensor_name(encoder.num_io_tensors - 1) == "encoder_output" or any(
        encoder.get_tensor_name(i) == "encoder_output" for i in range(encoder.num_io_tensors)
    )
    assert len(fp8_decoder) < len(plain_decoder)
    assert len(fp8_encoder) < len(plain_encoder)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""
        self.profiling_verbosity = None


@pytest.mark.parametrize(
    ("routes", "expected"),
    [('{"peep:match_dual_gemm": {}}', "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, expected: str) -> None:
    """The FP8 build route is applied only when the TensorRT build lists the knob."""
    ctx = WhisperQuantContext(scales={"layer.0.w_fc1": object()}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    model._configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
