# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary FP8 self-quantization: weights, scales, graph wiring and the calibration tool."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    CanaryQuantContext,
    calibrate_canary,
    configure_quantized_build,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, MEL, ML, VOCAB, ENC_LAYERS, DEC_LAYERS, HEADS, SUB, KERN = (
    128,
    256,
    16,
    40,
    64,
    2,
    2,
    4,
    8,
    3,
)
ES = 5  # encoder frames after the three stride-2 subsampling stages of ML = 40 mel frames
MAX_TGT = 16


def _tiny_weights(seed: int = 0) -> dict:
    """A small random weight dict with the keys ``load_weights`` produces."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm(size):
        return np.ones(size, dtype=np.float32)

    hd = D // HEADS
    w: dict = {
        "_enc_layers": ENC_LAYERS,
        "_dec_layers": DEC_LAYERS,
        "_enc_heads": HEADS,
        "_dec_heads": HEADS,
        "_enc_ffn": FFN,
        "_dec_ffn": FFN,
        "_hidden": D,
        "_vocab": VOCAB,
        "_mel_bins": MEL,
        "_kern": KERN,
        "_mel_length": ML,
        "_enc_seq": ES,
        "_sub_ch": SUB,
        "_head_dim": hd,
        "_max_tgt": MAX_TGT,
    }
    feat_out = MEL // 8
    w["enc_sub_conv0_w"], w["enc_sub_conv0_b"] = mat(SUB, 1, 3, 3), mat(SUB)
    for s in range(2):
        w[f"enc_sub_dw{s}_w"], w[f"enc_sub_dw{s}_b"] = mat(SUB, 1, 3, 3), mat(SUB)
        w[f"enc_sub_pw{s}_w"], w[f"enc_sub_pw{s}_b"] = mat(SUB, SUB, 1, 1), mat(SUB)
    w["enc_sub_out_w"], w["enc_sub_out_b"] = mat(SUB * feat_out, D), mat(D)
    for i in range(ENC_LAYERS):
        pk = f"el.{i}"
        for name in ("w_q", "w_k", "w_v", "w_o"):
            w[f"{pk}.{name}"] = mat(D, D)
            w[f"{pk}.b_{name[-1]}"] = mat(D)
        w[f"{pk}.pos_bias_u"], w[f"{pk}.pos_bias_v"] = mat(HEADS, hd), mat(HEADS, hd)
        w[f"{pk}.rpe_proj"] = mat(2 * ES - 1, HEADS, hd)
        w[f"{pk}.norm_sa"], w[f"{pk}.norm_sa_b"] = norm(D), mat(D)
        for ff in ("ff1", "ff2"):
            w[f"{pk}.{ff}.w1"], w[f"{pk}.{ff}.b1"] = mat(D, FFN), mat(FFN)
            w[f"{pk}.{ff}.w2"], w[f"{pk}.{ff}.b2"] = mat(FFN, D), mat(D)
            w[f"{pk}.{ff}.norm"], w[f"{pk}.{ff}.norm_b"] = norm(D), mat(D)
        w[f"{pk}.cpw1_w"], w[f"{pk}.cpw1_b"] = mat(2 * D, D, 1), mat(2 * D)
        w[f"{pk}.cdw_w"], w[f"{pk}.cdw_b"] = mat(D, 1, KERN), mat(D)
        w[f"{pk}.bn_w"], w[f"{pk}.bn_b"] = norm(D), mat(D)
        w[f"{pk}.bn_m"], w[f"{pk}.bn_v"] = mat(D), norm(D)
        w[f"{pk}.cpw2_w"], w[f"{pk}.cpw2_b"] = mat(D, D, 1), mat(D)
        w[f"{pk}.norm_conv"], w[f"{pk}.norm_conv_b"] = norm(D), mat(D)
        w[f"{pk}.norm_out"], w[f"{pk}.norm_out_b"] = norm(D), mat(D)
    w["dec_emb"], w["dec_pos"] = mat(VOCAB, D), mat(MAX_TGT, D)
    w["emb_ln"], w["emb_ln_b"] = norm(D), mat(D)
    for i in range(DEC_LAYERS):
        pk = f"layer.{i}"
        for name, bias in (
            ("w_q", "q_bias"),
            ("w_k", "k_bias"),
            ("w_v", "v_bias"),
            ("w_o", "o_bias"),
            ("xw_q", "xb_q"),
            ("xw_k", "xb_k"),
            ("xw_v", "xb_v"),
            ("xw_o", "xb_o"),
        ):
            w[f"{pk}.{name}"], w[f"{pk}.{bias}"] = mat(D, D), mat(D)
        w[f"{pk}.w_fc1"], w[f"{pk}.fc1_bias"] = mat(D, FFN), mat(FFN)
        w[f"{pk}.w_fc2"], w[f"{pk}.fc2_bias"] = mat(FFN, D), mat(D)
        for name in ("input_norm", "xattn_norm", "ffn_norm"):
            w[f"{pk}.{name}"], w[f"{pk}.{name}_b"] = norm(D), mat(D)
    w["final_norm"], w["final_norm_b"] = norm(D), mat(D)
    w["w_out"], w["out_bias"] = mat(D, VOCAB), mat(VOCAB)
    return w


def _all_scales(weights: dict, value: float = 0.02) -> dict[str, float]:
    return {name: value for name in projection_names(weights)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights with a scale for every projection in place of the committed table."""
    weights = _tiny_weights()
    scales = _all_scales(weights)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    return weights, scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight within e4m3 error."""
    weight = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    np.testing.assert_allclose(decoded, weight, rtol=0.07, atol=scale)


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax cannot be scaled."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_every_projection_is_quantized(tiny) -> None:
    """Encoder attention and feed-forward and every decoder projection get a scale."""
    weights, scales = tiny
    ctx = calibrate_canary(weights, graph_ops, "fp8")
    assert set(ctx.scales) == set(scales) - {
        f"layer.{i}.{name}" for i in range(DEC_LAYERS) for name in ("w_o", "xw_o")
    }
    assert len(ctx.scales) == ENC_LAYERS * 8 + DEC_LAYERS * 8
    # The decoder attention outputs follow the fused attention, which TensorRT 11.4 cannot
    # compile with an FP8 Quantize; the encoder attention is decomposed and is quantized.
    assert not ctx.should_quantize("layer.0.w_o") and not ctx.should_quantize("layer.1.xw_o")
    assert ctx.should_quantize("el.0.w_o")
    assert ctx.has_fp8_weights
    assert ctx.should_quantize("el.0.ff1.w1") and ctx.should_quantize("layer.1.xw_k")
    # Not projections that go through a matmul: the output head and the positional projection.
    assert not ctx.should_quantize("w_out") and not ctx.should_quantize("el.0.w_pos")


def test_projection_without_scale_stays_unquantized(tiny, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    weights, scales = tiny
    scales = dict(scales)
    del scales["layer.0.xw_k"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    ctx = calibrate_canary(weights, graph_ops, "fp8")
    assert not ctx.should_quantize("layer.0.xw_k")
    assert ctx.should_quantize("layer.0.xw_v")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_canary(weights, graph_ops, "nvfp4")


def test_unknown_model_shape_is_reported() -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    weights = _tiny_weights()
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_canary(weights, graph_ops, "fp8")


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"canary-h(\d+)-e(\d+)-d(\d+)-m(\d+)", key)
        assert match, key
        hidden, enc, dec, mel = (int(g) for g in match.groups())
        weights = {"_hidden": hidden, "_enc_layers": enc, "_dec_layers": dec, "_mel_bins": mel}
        assert scale_key(weights) == key
        expected = set(projection_names(weights))
        assert set(scales) == expected, key
        assert all(value > 0 for value in scales.values()), key


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps module paths of the reference model to weight names."""
    name = calibrate_fp8_scales.weight_name
    for module, expected in calibrate_fp8_scales.EXAMPLE_NAMES.items():
        assert name(module) == expected, module
    assert name("proj_out") is None
    assert name("decoder.layers.0.mlp.fc1.base") is None


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


def test_build_requires_fp16_without_overrides(tmp_path: Path) -> None:
    """FP8 builds refuse fp32, fp32 layers and tensor parallelism before loading anything."""
    base = dict(
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="transcription",
        model_dir=str(tmp_path),
        quantization="fp8",
        precision="fp16",
        fp32_layers=(),
        tensor_parallel_size=1,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "canary"}))
    for change, message in (
        ({"precision": "fp32"}, "fp16"),
        ({"fp32_layers": (3,)}, "fp32_layers"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
    ):
        with pytest.raises(ValueError, match=message):
            model.build(SimpleNamespace(**{**base, **change}), None)
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(SimpleNamespace(**{**base, "quantization": "nvfp4"}), None)


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
    ctx = CanaryQuantContext(scales={"layer.0.w_fc1": 0.02}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.profiling_verbosity is None


def _tiny_config() -> ModelConfig:
    return ModelConfig(model_type="canary", raw={"_fp32_layers": ()})


def test_quantized_engines_build(tiny) -> None:
    """The encoder and decoder engines build with FP8 projections and are smaller than fp16."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights, _ = tiny
    canary = model._CanaryModel()
    config = _tiny_config()
    ctx = calibrate_canary(weights, graph_ops, "fp8")

    plain_decoder = canary.build_engine(config, weights, 8, precision="fp16")
    plain_encoder = canary.build_vision_engine("", config, weights, precision="fp16")
    fp8_decoder = canary.build_engine(config, weights, 8, precision="fp16", quant_ctx=ctx)
    fp8_encoder = canary.build_vision_engine("", config, weights, precision="fp16", quant_ctx=ctx)

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    decoder = runtime.deserialize_cuda_engine(fp8_decoder)
    names = {decoder.get_tensor_name(i) for i in range(decoder.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "cross_k_0"} <= names
    encoder = runtime.deserialize_cuda_engine(fp8_encoder)
    assert "encoder_output" in {encoder.get_tensor_name(i) for i in range(encoder.num_io_tensors)}
    assert len(fp8_decoder) < len(plain_decoder)
    assert len(fp8_encoder) < len(plain_encoder)
