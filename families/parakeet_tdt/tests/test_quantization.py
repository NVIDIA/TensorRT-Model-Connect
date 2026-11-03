# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parakeet TDT FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.parakeet_tdt import (  # noqa: E402
    calibrate_fp8_scales,
    engines,
    fastconformer,
    graph_ops,
    model,
)
from families.parakeet_tdt import quantization as quantization_module  # noqa: E402
from families.parakeet_tdt.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    ParakeetQuantContext,
    calibrate_parakeet,
    configure_fp8_build,
    projections,
    quantize_fp8_weight,
    reject_prequantized,
    scale_key,
)
from tensorrt_model_connect.build import BuildRequest  # noqa: E402
from tensorrt_model_connect.bundle_writer import BundleWriter  # noqa: E402

# Tiny FastConformer: hidden D with HEADS heads, MEL bins and MEL_LENGTH frames. Three
# stride-2 subsampling stages give ENC_SEQ = 8 encoder frames and a feature width of 2.
D, FFN, HEADS, KERNEL, MEL, MEL_LENGTH, SUB_CH, LAYERS = 64, 128, 4, 3, 16, 64, 8, 2
ENC_SEQ = fastconformer._compute_enc_seq_len(MEL_LENGTH)
CONFIG = SimpleNamespace(
    model_type="parakeet_tdt",
    encoder_hidden_size=D,
    encoder_layers=LAYERS,
    encoder_ffn_size=FFN,
    num_mel_bins=MEL,
)


def _tiny_weights() -> dict:
    """Random encoder weights in the family weight-dict layout (matrices are [in, out])."""
    rng = np.random.default_rng(0)
    head_dim = D // HEADS

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights: dict = {
        "_enc_layers": LAYERS,
        "_enc_heads": HEADS,
        "_hidden": D,
        "_head_dim": head_dim,
        "_enc_ffn": FFN,
        "_kern": KERNEL,
        "_mel_bins": MEL,
        "_mel_length": MEL_LENGTH,
        "_enc_seq": ENC_SEQ,
        "_sub_ch": SUB_CH,
        "enc_sub_conv0_w": w(SUB_CH, 1, 3, 3),
        "enc_sub_conv0_b": w(SUB_CH),
        "enc_sub_out_w": w(SUB_CH * 2, D),
        "enc_sub_out_b": w(D),
    }
    for stage in range(2):
        weights[f"enc_sub_dw{stage}_w"] = w(SUB_CH, 1, 3, 3)
        weights[f"enc_sub_dw{stage}_b"] = w(SUB_CH)
        weights[f"enc_sub_pw{stage}_w"] = w(SUB_CH, SUB_CH, 1, 1)
        weights[f"enc_sub_pw{stage}_b"] = w(SUB_CH)
    for layer in range(LAYERS):
        pfx = f"el.{layer}"
        for stem in ("q", "k", "v", "o"):
            weights[f"{pfx}.w_{stem}"] = w(D, D)
            weights[f"{pfx}.b_{stem}"] = w(D)
        weights[f"{pfx}.pos_bias_u"] = w(HEADS, head_dim)
        weights[f"{pfx}.pos_bias_v"] = w(HEADS, head_dim)
        weights[f"{pfx}.rpe_proj"] = w(2 * ENC_SEQ - 1, HEADS, head_dim)
        for norm in ("norm_sa", "norm_conv", "norm_out"):
            weights[f"{pfx}.{norm}"] = np.ones(D, np.float32)
            weights[f"{pfx}.{norm}_b"] = np.zeros(D, np.float32)
        for half in ("ff1", "ff2"):
            weights[f"{pfx}.{half}.w1"] = w(D, FFN)
            weights[f"{pfx}.{half}.b1"] = w(FFN)
            weights[f"{pfx}.{half}.w2"] = w(FFN, D)
            weights[f"{pfx}.{half}.b2"] = w(D)
            weights[f"{pfx}.{half}.norm"] = np.ones(D, np.float32)
            weights[f"{pfx}.{half}.norm_b"] = np.zeros(D, np.float32)
        weights[f"{pfx}.cpw1_w"] = w(2 * D, D, 1)
        weights[f"{pfx}.cpw1_b"] = w(2 * D)
        weights[f"{pfx}.cdw_w"] = w(D, 1, KERNEL)
        weights[f"{pfx}.cdw_b"] = w(D)
        weights[f"{pfx}.bn_w"] = np.ones(D, np.float32)
        weights[f"{pfx}.bn_b"] = np.zeros(D, np.float32)
        weights[f"{pfx}.bn_m"] = np.zeros(D, np.float32)
        weights[f"{pfx}.bn_v"] = np.ones(D, np.float32)
        weights[f"{pfx}.cpw2_w"] = w(D, D, 1)
        weights[f"{pfx}.cpw2_b"] = w(D)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for _, name in projections(CONFIG)}


@pytest.fixture
def all_scales(monkeypatch):
    """Use an activation scale for every projection of the tiny model."""
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda config: scales)
    return scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The weight scale is amax / 448 and the FP8 bytes round-trip within e4m3 precision."""
    weight = np.random.default_rng(1).standard_normal((32, 48)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= 0.07 * np.abs(weight).max()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax has no scale."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), np.float32))


def test_every_encoder_projection_is_quantized(all_scales) -> None:
    """Both feed-forward halves and q/k/v/o of every layer get FP8 weights in [out, in]."""
    ctx = calibrate_parakeet(_tiny_weights(), graph_ops, "fp8", CONFIG)
    assert len(ctx.scales) == LAYERS * 8
    assert ctx.scales["el.1.ff2.w1"].weight.packed.shape == (FFN, D)
    assert ctx.scales["el.0.ff1.w2"].weight.packed.shape == (D, FFN)
    for stem in ("w_q", "w_k", "w_v", "w_o"):
        assert ctx.scales[f"el.1.{stem}"].weight.packed.shape == (D, D)
        assert ctx.should_quantize(f"el.0.{stem}")
    # Convolution module, relative-position projection and subsampling stem stay fp16.
    for name in ("el.0.cpw1_w", "el.0.cpw2_w", "el.0.cdw_w", "el.0.w_pos", "enc_sub_out_w"):
        assert not ctx.should_quantize(name)


def test_projection_without_scale_is_an_error(all_scales) -> None:
    """A projection with no activation scale is an error, not a silent skip."""
    del all_scales["el.0.ff1.w1"]
    with pytest.raises(ValueError, match=r"el\.0\.ff1\.w1.*calibrate_fp8_scales"):
        calibrate_parakeet(_tiny_weights(), graph_ops, "fp8", CONFIG)


@pytest.mark.parametrize("name", ["el.1.w_q", "el.0.w_o"])
def test_attention_projection_without_scale_is_an_error(all_scales, name: str) -> None:
    """An attention projection with no activation scale is an error, not a silent fp16 fallback."""
    del all_scales[name]
    with pytest.raises(ValueError, match=rf"{re.escape(name)}.*calibrate_fp8_scales"):
        calibrate_parakeet(_tiny_weights(), graph_ops, "fp8", CONFIG)


def test_missing_weight_is_reported(all_scales) -> None:
    """A weight dict without a scaled projection is an error, not a silent skip."""
    weights = _tiny_weights()
    del weights["el.1.ff2.w2"]
    with pytest.raises(ValueError, match="el.1.ff2.w2"):
        calibrate_parakeet(weights, graph_ops, "fp8", CONFIG)


def test_already_quantized_checkpoint_is_rejected(monkeypatch) -> None:
    """A checkpoint that already stores quantization scales is not quantized a second time."""
    reject_prequantized({"encoder.layers.0.self_attn.q_proj.weight": 1})
    state = {"encoder.layers.0.feed_forward1.linear1.weight_scale": 1}
    with pytest.raises(ValueError, match="already quantized"):
        reject_prequantized(state)
    monkeypatch.setattr(engines, "_load_hf_as_nemo", lambda model_dir: (state, {}))
    with pytest.raises(ValueError, match="already quantized"):
        engines._load_weights("unused", None, quantization="fp8")


def test_unsupported_format_is_rejected() -> None:
    """Only fp8 is implemented."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_parakeet(_tiny_weights(), graph_ops, "nvfp4", CONFIG)


def test_unknown_model_config_is_reported() -> None:
    """A model config with no committed scales raises and names the calibration tool."""
    other = replace_config(CONFIG, encoder_hidden_size=D + 1)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_parakeet(_tiny_weights(), graph_ops, "fp8", other)


def replace_config(config, **changes):
    """Copy of a namespace config with some fields changed."""
    return SimpleNamespace(**{**vars(config), **changes})


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has a positive scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"parakeet_tdt-h(\d+)-e(\d+)-f(\d+)-m(\d+)", key)
        assert match, key
        hidden, layers, ffn, mels = (int(g) for g in match.groups())
        config = SimpleNamespace(
            model_type="parakeet_tdt",
            encoder_hidden_size=hidden,
            encoder_layers=layers,
            encoder_ffn_size=ffn,
            num_mel_bins=mels,
        )
        assert scale_key(config) == key
        assert {name for _, name in projections(config)} == set(scales), key
        assert all(np.isfinite(value) and value > 0 for value in scales.values()), key


def test_calibration_tool_maps_module_names_to_weight_names() -> None:
    """The calibration tool maps encoder module paths to internal weight names."""
    name = calibrate_fp8_scales.weight_name
    assert name("encoder.layers.0.feed_forward2.linear1") == "el.0.ff2.w1"
    assert name("layers.4.self_attn.o_proj") == "el.4.w_o"
    assert name("encoder.layers.0.self_attn.q_proj") == "el.0.w_q"
    assert name("layers.11.self_attn.k_proj") == "el.11.w_k"
    assert name("layers.23.self_attn.v_proj") == "el.23.w_v"
    assert name("layers.2.self_attn.q_proj.base") is None
    assert name("layers.7.feed_forward1.linear1") == "el.7.ff1.w1"
    assert name("layers.23.feed_forward2.linear2") == "el.23.ff2.w2"
    assert name("layers.3.self_attn.relative_k_proj") is None
    assert name("layers.3.conv.pointwise_conv1") is None
    assert name("layers.1.feed_forward1.linear1.base") is None
    assert name("subsampling.linear") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_clips", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, c: seen.update(calibrate=(d, c))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "clips")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--clips", "clips"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "clips", 96)
    assert seen["calibrate"] == (tmp_path, Path("clips"))


def _request(tmp_path: Path, **changes) -> BuildRequest:
    base = BuildRequest(
        model_dir=tmp_path / "checkpoint",
        output_path=tmp_path / "model.bundle",
        family="parakeet_tdt",
        task="speech_transcription",
        precision="fp16",
    )
    return replace(base, **changes)


def test_build_requires_fp16_for_fp8(tmp_path: Path) -> None:
    """FP8 builds are fp16 only and fail before the checkpoint is read."""
    with pytest.raises(ValueError, match="precision=fp16"):
        model.build(
            _request(tmp_path, precision="fp32", quantization="fp8"),
            BundleWriter(tmp_path / "bad.bundle"),
        )


def test_build_passes_fp8_to_the_engine_compiler(tmp_path: Path, monkeypatch) -> None:
    """quantization=fp8 reaches compile_engines; unquantized builds do not pass the argument."""
    request = _request(tmp_path, quantization="fp8")
    request.model_dir.mkdir()
    fixture = Path(__file__).with_name("config.json")
    (request.model_dir / "config.json").write_bytes(fixture.read_bytes())
    (request.model_dir / "model.safetensors").write_bytes(b"weights validated elsewhere")
    (request.model_dir / "tokenizer.json").write_text('{"model":{"type":"Unigram"}}')
    seen = []

    def compile_engines(model_dir, *, precision, verbose, **kwargs):
        seen.append(kwargs)
        return {
            "encoder.plan": b"e",
            "predictor.plan": b"p",
            "joint.plan": b"j",
            "mel_filterbank": b"m",
        }, {"tdt_blank_id": 8192}

    monkeypatch.setitem(
        sys.modules,
        "families.parakeet_tdt.engines",
        SimpleNamespace(compile_engines=compile_engines),
    )
    model.build(request, BundleWriter(request.output_path))
    model.build(replace(request, quantization=None), BundleWriter(tmp_path / "plain.bundle"))
    assert seen == [{"quantization": "fp8"}, {}]


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
    ctx = ParakeetQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _run(plan: bytes, mel: np.ndarray, mask: np.ndarray):
    """Run an encoder plan on the GPU and return its output."""
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    mel_t = torch.from_numpy(mel).cuda()
    mask_t = torch.from_numpy(mask).cuda()
    out = torch.empty(tuple(engine.get_tensor_shape("encoder_output")), device="cuda")
    for name, tensor in (
        ("mel_features", mel_t),
        ("encoder_mask", mask_t),
        ("encoder_output", out),
    ):
        context.set_tensor_address(name, tensor.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.cpu().numpy()


def test_quantized_encoder_builds_and_matches_fp16(all_scales, monkeypatch) -> None:
    """The FP8 encoder builds, routes every feed-forward and attention projection through FP8 Q/DQ and tracks fp16."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights = _tiny_weights()
    ctx = calibrate_parakeet(weights, graph_ops, "fp8", CONFIG)
    routed = []
    original = quantization_module._wrap_fp8_matmul
    monkeypatch.setattr(
        quantization_module,
        "_wrap_fp8_matmul",
        lambda *args, **kwargs: routed.append(1) or original(*args, **kwargs),
    )
    plain = fastconformer._build_encoder(None, weights, precision="fp16")
    fp8 = fastconformer._build_encoder(None, weights, precision="fp16", quant_ctx=ctx)
    assert len(routed) == LAYERS * 8

    rng = np.random.default_rng(2)
    mel = rng.standard_normal((MEL, MEL_LENGTH)).astype(np.float32)
    mask = np.zeros((1, 1, ENC_SEQ), np.float32)
    reference, quantized = _run(plain, mel, mask), _run(fp8, mel, mask)
    assert np.isfinite(quantized).all()
    cosine = float(
        (reference * quantized).sum() / (np.linalg.norm(reference) * np.linalg.norm(quantized))
    )
    assert cosine > 0.95, cosine
