# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Speech Streaming FP8 self-quantization: weights, scales, graph wiring and tools."""

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
from ..canary_encoder_helpers import _build_encoder  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    NemotronQuantContext,
    calibrate_nemotron,
    configure_quantized_build,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, MEL, VOCAB, LAYERS, HEADS, SUB, KERN = 128, 256, 16, 64, 2, 4, 8, 9
CACHE_LEFT = 4
RIGHT = 1
MEL_LEN, ENC_SEQ = 25, 4  # offline plan: 25 mel frames give 4 encoder frames (causal subsampling)


def _tiny_weights(seed: int = 0) -> dict:
    """A small random weight dict with the keys ``load_weights`` produces."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm(size):
        return np.ones(size, dtype=np.float32)

    hd = D // HEADS
    feat = MEL
    for _ in range(3):
        feat = feat // 2 + 1  # causal subsampling stage on the feature axis
    w: dict = {
        "_enc_layers": LAYERS,
        "_enc_heads": HEADS,
        "_enc_ffn": FFN,
        "_hidden": D,
        "_vocab": VOCAB,
        "_mel_bins": MEL,
        "_kern": KERN,
        "_mel_length": MEL_LEN,
        "_enc_seq": ENC_SEQ,
        "_sub_ch": SUB,
        "_head_dim": hd,
        "_conv_norm_type": "batch_norm",
        "_conv_context_size": "symmetric",
        "_causal_downsampling": True,
        "_encoder_attention_mask_2d": True,
        "_streaming_cache_left": CACHE_LEFT,
        "_streaming_right_contexts": [RIGHT],
    }
    w["enc_sub_conv0_w"], w["enc_sub_conv0_b"] = mat(SUB, 1, 3, 3), mat(SUB)
    for s in range(2):
        w[f"enc_sub_dw{s}_w"], w[f"enc_sub_dw{s}_b"] = mat(SUB, 1, 3, 3), mat(SUB)
        w[f"enc_sub_pw{s}_w"], w[f"enc_sub_pw{s}_b"] = mat(SUB, SUB, 1, 1), mat(SUB)
    w["enc_sub_out_w"], w["enc_sub_out_b"] = mat(SUB * feat, D), mat(D)
    for i in range(LAYERS):
        pk = f"el.{i}"
        for name in ("w_q", "w_k", "w_v", "w_o"):
            w[f"{pk}.{name}"] = mat(D, D)
            w[f"{pk}.b_{name[-1]}"] = mat(D)
        w[f"{pk}.pos_bias_u"], w[f"{pk}.pos_bias_v"] = mat(HEADS, hd), mat(HEADS, hd)
        w[f"{pk}.w_pos"] = mat(D, D)
        w[f"{pk}.rpe_proj"] = mat(2 * ENC_SEQ - 1, HEADS, hd)
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


def test_every_encoder_projection_is_quantized(tiny) -> None:
    """Attention q/k/v/o and the four feed-forward matrices of every layer get a scale."""
    weights, scales = tiny
    ctx = calibrate_nemotron(weights, graph_ops, "fp8")
    assert set(ctx.scales) == set(scales)
    assert len(ctx.scales) == LAYERS * 8
    # The relative-position attention is decomposed, so its output projection is quantized.
    assert ctx.should_quantize("el.0.w_o") and ctx.should_quantize("el.1.ff2.w1")
    assert ctx.has_fp8_weights
    # Not projections that go through the scale table: the positional projection and the convolutions.
    assert not ctx.should_quantize("el.0.w_pos") and not ctx.should_quantize("el.0.cpw1_w")


def test_projection_without_scale_stays_unquantized(tiny, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    weights, scales = tiny
    scales = dict(scales)
    del scales["el.0.ff1.w1"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    ctx = calibrate_nemotron(weights, graph_ops, "fp8")
    assert not ctx.should_quantize("el.0.ff1.w1")
    assert ctx.should_quantize("el.0.ff1.w2")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_nemotron(weights, graph_ops, "nvfp4")


def test_unknown_checkpoint_is_reported() -> None:
    """A checkpoint with no committed scales raises and names the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_nemotron(_tiny_weights(), graph_ops, "fp8")


def test_scale_key_names_shape_and_vocabulary() -> None:
    """The English and the multilingual checkpoints share a shape but not a key."""
    weights = _tiny_weights()
    assert scale_key(weights) == f"nemotron_speech_streaming-h{D}-e{LAYERS}-m{MEL}-v{VOCAB}"
    other = {**weights, "_vocab": VOCAB + 1}
    assert scale_key(other) != scale_key(weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed checkpoint has a scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"nemotron_speech_streaming-h(\d+)-e(\d+)-m(\d+)-v(\d+)", key)
        assert match, key
        hidden, layers, mel, vocab = (int(g) for g in match.groups())
        weights = {"_hidden": hidden, "_enc_layers": layers, "_mel_bins": mel, "_vocab": vocab}
        assert scale_key(weights) == key
        assert set(scales) == set(projection_names(weights)), key
        assert all(value > 0 for value in scales.values()), key


def test_calibration_context_adds_one_amax_output_per_projection() -> None:
    """The calibration stand-in keeps the plain matmul and marks max(abs(input)) as an output."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    context = calibrate_fp8_scales.CalibrationContext(graph_ops)
    x = network.add_input("x", trt.float32, (5, 8))
    out = context.maybe_quantized_matmul(
        network, x, 8, 4, np.ones((8, 4), dtype=np.float32), "el.0.ff1.w1"
    )
    assert tuple(out.shape) == (5, 4)
    assert context.outputs == {"el.0.ff1.w1": "amax_el_0_ff1_w1"}
    assert network.get_output(0).name == "amax_el_0_ff1_w1"


def test_mel_spectrogram_matches_the_front_end_geometry() -> None:
    """One second of audio gives 100 frames of the filterbank's width; silence is the log guard."""
    pytest.importorskip("torch")
    filters = np.ones((257, MEL), dtype=np.float32) / 257
    mel = calibrate_fp8_scales.mel_spectrogram(np.zeros(16000, dtype=np.float32), filters)
    assert mel.shape == (MEL, 100)
    assert np.allclose(mel, np.log(2.0**-24), atol=1e-4)
    tone = np.sin(np.arange(16000, dtype=np.float32) * 0.2)
    assert calibrate_fp8_scales.mel_spectrogram(tone, filters).mean() > mel.mean()


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_clips", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, c, r: seen.update(calibrate=(d, c, r))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "clips")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--clips", "clips"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "clips", 96)
    assert seen["calibrate"] == (tmp_path, Path("clips"), 13)


def test_build_requires_fp16_without_tensor_parallelism(tmp_path: Path) -> None:
    """FP8 builds refuse fp32 and tensor parallelism."""
    base = dict(
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="transcription_streaming",
        model_dir=str(tmp_path),
        quantization="fp8",
        precision="fp16",
        fp32_layers=(),
        tensor_parallel_size=1,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_speech_streaming"}))
    for change, message in (
        ({"precision": "fp32"}, "fp16"),
        ({"tensor_parallel_size": 2}, "tensor-parallel"),
    ):
        with pytest.raises(ValueError, match=message):
            model.build(SimpleNamespace(**{**base, **change}), None)


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
    ctx = NemotronQuantContext(scales={"el.0.ff1.w1": 0.02}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.profiling_verbosity is None


def test_quantized_engines_build(tiny) -> None:
    """The offline and both streaming encoder plans build with FP8 projections and shrink."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights, _ = tiny
    config = ModelConfig(model_type="nemotron_speech_streaming", raw={})
    ctx = calibrate_nemotron(weights, graph_ops, "fp8")

    plain_offline = _build_encoder(config, weights, precision="fp16")
    fp8_offline = _build_encoder(config, weights, precision="fp16", quant_ctx=ctx)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    offline = runtime.deserialize_cuda_engine(fp8_offline)
    assert "encoder_output" in {offline.get_tensor_name(i) for i in range(offline.num_io_tensors)}
    assert len(fp8_offline) < len(plain_offline)

    for first_step in (True, False):
        plain = model._build_streaming_encoder(
            weights, RIGHT, first_step=first_step, precision="fp16"
        )
        fp8 = model._build_streaming_encoder(
            weights, RIGHT, first_step=first_step, precision="fp16", quant_ctx=ctx
        )
        engine = runtime.deserialize_cuda_engine(fp8)
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        assert {"encoder_output", "cache_last_channel_next", "cache_last_time_next"} <= names
        assert len(fp8) < len(plain)
