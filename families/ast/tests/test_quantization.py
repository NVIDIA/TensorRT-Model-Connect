# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AST FP8 self-quantization: weights, scales, graph wiring and the calibration tool."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    HF_SUFFIX,
    AstQuantContext,
    calibrate_ast,
    configure_quantized_build,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, LABELS = 128, 256, 2, 4, 10
MEL, LENGTH, PATCH = 8, 16, 4
PREFIX = "audio_spectrogram_transformer"
CFG = {
    "hidden_size": D,
    "num_layers": LAYERS,
    "num_heads": HEADS,
    "head_dim": D // HEADS,
    "eps": 1e-12,
    "patch": PATCH,
    "frequency_stride": PATCH,
    "time_stride": PATCH,
    "frequency_patches": (MEL - PATCH) // PATCH + 1,
    "time_patches": (LENGTH - PATCH) // PATCH + 1,
    "num_patches": 8,
    "seq_len": 10,
    "num_labels": LABELS,
    "num_mel_bins": MEL,
    "max_length": LENGTH,
}


def _tiny_weights(seed: int = 0) -> dict:
    """A small random checkpoint with the tensors ``model._load`` returns."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm(size):
        return np.ones(size, dtype=np.float32)

    w: dict = {
        f"{PREFIX}.embeddings.cls_token": mat(1, 1, D),
        f"{PREFIX}.embeddings.distillation_token": mat(1, 1, D),
        f"{PREFIX}.embeddings.position_embeddings": mat(1, CFG["seq_len"], D),
        f"{PREFIX}.embeddings.patch_embeddings.projection.weight": mat(D, 1, PATCH, PATCH),
        f"{PREFIX}.embeddings.patch_embeddings.projection.bias": mat(D),
        f"{PREFIX}.layernorm.weight": norm(D),
        f"{PREFIX}.layernorm.bias": mat(D),
        "classifier.layernorm.weight": norm(D),
        "classifier.layernorm.bias": mat(D),
        "classifier.dense.weight": mat(LABELS, D),
        "classifier.dense.bias": mat(LABELS),
    }
    for i in range(LAYERS):
        layer = f"{PREFIX}.encoder.layer.{i}"
        for stem, suffix in HF_SUFFIX.items():
            out, inner = (FFN, D) if stem == "fc1" else (D, FFN) if stem == "fc2" else (D, D)
            w[f"{layer}.{suffix}.weight"] = mat(out, inner)
            w[f"{layer}.{suffix}.bias"] = mat(out)
        for name in ("layernorm_before", "layernorm_after"):
            w[f"{layer}.{name}.weight"], w[f"{layer}.{name}.bias"] = norm(D), mat(D)
    return w


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in projection_names(CFG)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights with a scale for every projection in place of the committed table."""
    weights = _tiny_weights()
    scales = _all_scales()
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
    """The six projections of every layer get a scale; the head and the patch stem do not."""
    weights, scales = tiny
    ctx = calibrate_ast(weights, CFG, graph, "fp8")
    assert set(ctx.scales) == set(scales)
    assert len(ctx.scales) == LAYERS * 6
    # The attention is decomposed, so its output projection is quantized as well.
    assert ctx.should_quantize("layer.0.attn_out") and ctx.should_quantize("layer.1.fc2")
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("classifier") and not ctx.should_quantize("layer.0.patch")


def test_projection_without_scale_is_rejected(tiny, monkeypatch) -> None:
    """A projection with no activation scale fails loudly and names the calibration tool."""
    weights, scales = tiny
    scales = dict(scales)
    del scales["layer.0.fc1"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.fc1.*calibrate_fp8_scales"):
        calibrate_ast(weights, CFG, graph, "fp8")


def test_quantized_checkpoint_is_rejected(tiny) -> None:
    """An already quantized checkpoint raises instead of being quantized twice."""
    weights, _ = tiny
    weights[f"{PREFIX}.encoder.layer.0.intermediate.dense.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_ast(weights, CFG, graph, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_ast(weights, CFG, graph, "nvfp4")


def test_unknown_checkpoint_is_reported() -> None:
    """A checkpoint with no committed scales raises and names the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_ast(_tiny_weights(), CFG, graph, "fp8")


def test_scale_key_separates_checkpoints_of_one_geometry() -> None:
    """Two AudioSet releases of one shape get different keys through the classifier bias."""
    weights = _tiny_weights()
    key = scale_key(CFG, weights)
    assert re.fullmatch(rf"ast-h{D}-l{LAYERS}-s10-[-+0-9._]+", key)
    other = {**weights, "classifier.dense.bias": weights["classifier.dense.bias"] + 1.0}
    assert scale_key(CFG, other) != key


def test_committed_scales_cover_every_projection() -> None:
    """Every committed checkpoint has a scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"ast-h(\d+)-l(\d+)-s(\d+)-[-+0-9._]+", key)
        assert match, key
        cfg = {"hidden_size": int(match[1]), "num_layers": int(match[2])}
        assert set(scales) == set(projection_names(cfg)), key
        assert all(value > 0 for value in scales.values()), key


def test_calibration_tool_maps_module_names_to_short_names() -> None:
    """The calibration tool maps module paths of the reference model to short names."""
    name = calibrate_fp8_scales.weight_name
    layer = f"{PREFIX}.layers.3"
    assert name(f"{layer}.attention.k_proj") == "layer.3.key"
    assert name(f"{layer}.attention.o_proj") == "layer.3.attn_out"
    assert name(f"{layer}.mlp.fc1") == "layer.3.fc1"
    assert name(f"{layer}.mlp.fc2") == "layer.3.fc2"
    assert name("classifier.dense") is None
    assert name(f"{layer}.attention") is None


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
    assert seen["fetch"] == (tmp_path / "clips", 256)
    assert seen["calibrate"] == (tmp_path, Path("clips"))


def test_build_requires_fp16(tmp_path: Path) -> None:
    """FP8 builds refuse fp32 before reading the checkpoint."""
    request = SimpleNamespace(
        task="audio_to_class_scores",
        backend="trt",
        dynamic_kv_cache=False,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        video_num_frames=None,
        image_height=None,
        image_width=None,
        quantization="fp8",
        fp32_layers=(),
        mtp_seq_len=None,
        precision="fp32",
        model_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="fp16"):
        model.build(request, None)


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
    ctx = AstQuantContext(scales={"layer.0.fc1": 0.02}, graph=graph)
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.profiling_verbosity is None


def test_quantized_engine_builds(tiny) -> None:
    """The engine builds with FP8 projections, is smaller than fp16 and keeps its IO."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights, _ = tiny
    ctx = calibrate_ast(weights, CFG, graph, "fp8")
    plain = model.build_engine(weights, CFG, precision="fp16")
    fp8 = model.build_engine(weights, CFG, precision="fp16", quant_ctx=ctx)
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8)
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"input_values", "logits"} <= names
    assert len(fp8) < len(plain)


def test_quantized_projection_dequantize_and_matmul_run_in_fp16() -> None:
    """Both Dequantize layers and the matmul of a quantized projection are fp16, with no cast."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("creating a TensorRT builder needs a GPU")
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(0)
    inp = network.add_input("x", trt.float16, (1, 4, D))
    ctx = AstQuantContext(scales={"layer.0.fc1": 0.02}, graph=graph)
    weight = _tiny_weights()[f"{PREFIX}.encoder.layer.0.intermediate.dense.weight"]
    out = ctx.linear(network, inp, "layer.0.fc1", weight, None, np.float16)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    dequant = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequant) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequant)
    matmul = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    assert len(matmul) == 1 and matmul[0].get_output(0).dtype == trt.float16
    assert not any(layer.type == trt.LayerType.CAST for layer in layers)
    assert out.dtype == trt.float16
