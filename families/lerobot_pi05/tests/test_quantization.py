# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pi 0.5 FP8 self-quantization: weights, scales, graph wiring and the calibration tool."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales
from .. import quantization as quantization_module
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    EXPERT_ROOT,
    PREFIX_ROOT,
    Pi05QuantContext,
    build_quant_context,
    configure_quantized_build,
    projection_names,
    quantize_fp8_weight,
    round_up_to,
    scale_key,
)

CFG = SimpleNamespace(camera_slots=3, tokenizer_max_length=200, chunk_size=50)
WIDTH, MLP = 32, 64  # tiny stand-ins for both towers


def _weights(seed: int = 0, width: int = WIDTH, mlp: int = MLP) -> dict:
    """Random weights for every projection, square except the MLP."""
    rng = np.random.default_rng(seed)
    weights = {"action_out_proj.bias": rng.standard_normal(32).astype(np.float32)}
    for name in projection_names():
        shape = (
            (mlp, width)
            if name.endswith(("gate_proj", "up_proj"))
            else ((width, mlp) if name.endswith("down_proj") else (width, width))
        )
        weights[f"{name}.weight"] = (rng.standard_normal(shape) * 0.05).astype(np.float32)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float | None]:
    return {name: value for name in projection_names()}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights with a scale for every projection in place of the committed table."""
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    return _weights(), scales


def test_fp8_weight_scale_is_amax_over_448_rounded_up_to_bf16() -> None:
    weight = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight, "bf16")
    exact = float(np.abs(weight).max()) / 448.0
    assert scale >= exact and scale == pytest.approx(exact, rel=2**-7)
    assert float(np.asarray(scale, ml_dtypes.bfloat16)) == scale
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    np.testing.assert_allclose(decoded, weight, rtol=0.07, atol=scale)
    _, fp32_scale = quantize_fp8_weight(weight, "fp32")
    assert fp32_scale == pytest.approx(exact, rel=1e-6)


def test_round_up_never_lowers_the_scale() -> None:
    for value in (1e-4, 0.0123456, 3.3):
        assert round_up_to(value, ml_dtypes.bfloat16) >= value
        assert round_up_to(value, np.float32) >= np.float32(value)


def test_fp8_weight_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_projection_list_covers_both_towers() -> None:
    names = projection_names()
    # gate/up/down of the 17 prefix layers with an MLP and of the 18 expert layers.
    assert len(names) == 17 * 3 + 18 * 3
    assert f"{PREFIX_ROOT}.16.mlp.down_proj" in names
    assert f"{PREFIX_ROOT}.17.mlp.down_proj" not in names
    assert f"{EXPERT_ROOT}.17.mlp.down_proj" in names
    # Attention stays in the engine precision.
    assert not any("self_attn" in name for name in names)


def test_every_projection_is_quantized(tiny) -> None:
    weights, scales = tiny
    ctx = build_quant_context(weights, CFG, "fp8", precision="bf16")
    assert set(ctx.scales) == set(scales) and ctx.has_fp8_weights
    assert not ctx.should_quantize("action_out_proj")
    assert not ctx.should_quantize(f"{EXPERT_ROOT}.0.self_attn.q_proj")
    assert not ctx.should_quantize(
        "paligemma.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1"
    )


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    weights, scales = tiny
    scales = {**scales, f"{PREFIX_ROOT}.3.mlp.down_proj": None}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    ctx = build_quant_context(weights, CFG, "fp8", precision="bf16")
    assert not ctx.should_quantize(f"{PREFIX_ROOT}.3.mlp.down_proj")
    assert ctx.should_quantize(f"{PREFIX_ROOT}.3.mlp.up_proj")
    assert len(ctx.scales) == len(projection_names()) - 1


def test_projection_without_scale_is_rejected(tiny, monkeypatch) -> None:
    weights, scales = tiny
    scales = dict(scales)
    del scales[f"{EXPERT_ROOT}.0.mlp.gate_proj"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match=r"does not match.*calibrate_fp8_scales"):
        build_quant_context(weights, CFG, "fp8", precision="bf16")


def test_bad_scale_value_is_rejected(tiny, monkeypatch) -> None:
    weights, scales = tiny
    scales = {**scales, f"{EXPERT_ROOT}.0.mlp.gate_proj": 0.0}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match="finite and positive"):
        build_quant_context(weights, CFG, "fp8", precision="bf16")


def test_quantized_checkpoint_is_rejected(tiny) -> None:
    weights, _ = tiny
    weights[f"{EXPERT_ROOT}.0.mlp.up_proj.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        build_quant_context(weights, CFG, "fp8", precision="bf16")


def test_fp32_precision_is_rejected(tiny) -> None:
    weights, _ = tiny
    with pytest.raises(ValueError, match="needs precision bf16"):
        build_quant_context(weights, CFG, "fp8", precision="fp32")


def test_unsupported_format_is_rejected(tiny) -> None:
    weights, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        build_quant_context(weights, CFG, "nvfp4", precision="bf16")


def test_unknown_checkpoint_is_reported() -> None:
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        build_quant_context(_weights(), CFG, "fp8", precision="bf16")


def test_scale_key_separates_checkpoints_of_one_geometry() -> None:
    weights = _weights()
    key = scale_key(CFG, weights)
    assert re.fullmatch(r"pi05-gemma_2b-gemma_300m-s3-p200-c50-[-+0-9._]+", key)
    other = {**weights, "action_out_proj.bias": weights["action_out_proj.bias"] + 0.5}
    assert scale_key(CFG, other) != key
    assert (
        scale_key(SimpleNamespace(camera_slots=2, tokenizer_max_length=200, chunk_size=50), weights)
        != key
    )


def test_committed_scales_cover_every_projection() -> None:
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        assert re.fullmatch(r"pi05-gemma_2b-gemma_300m-s\d+-p\d+-c\d+-[-+0-9._]+", key), key
        assert set(scales) == set(projection_names()), key
        assert all(value is None or value > 0 for value in scales.values()), key


def test_calibration_tool_maps_module_names() -> None:
    name = calibrate_fp8_scales.projection_name
    assert (
        name("model.paligemma_with_expert.paligemma.model.language_model.layers.4.mlp.up_proj")
        == f"{PREFIX_ROOT}.4.mlp.up_proj"
    )
    assert (
        name("model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj")
        == f"{EXPERT_ROOT}.0.mlp.down_proj"
    )
    assert (
        name(
            "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1"
        )
        is None
    )
    assert name("model.action_out_proj") is None
    assert name("model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj") is None
    assert (
        name("model.paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense")
        is None
    )


def test_calibration_split_and_outliers() -> None:
    assert (
        calibrate_fp8_scales.split_of(3) == "calib" and calibrate_fp8_scales.split_of(13) == "eval"
    )
    # The test fixture episodes are in neither set.
    assert all(calibrate_fp8_scales.split_of(e) is None for e in (0, 379, 807))
    scales = {f"{EXPERT_ROOT}.{i}.mlp.down_proj": 1.0 for i in range(5)}
    scales[f"{EXPERT_ROOT}.4.mlp.down_proj"] = 21.0
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked[f"{EXPERT_ROOT}.4.mlp.down_proj"] is None
    assert marked[f"{EXPERT_ROOT}.0.mlp.down_proj"] == 1.0


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_frames", lambda out: seen.update(fetch=out))
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda *args: seen.update(calibrate=args)
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path)])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", "m", "--tokenizer-dir", "t", "--frames", "f"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == tmp_path
    assert seen["calibrate"] == (Path("m"), Path("t"), Path("f"), None)


class _FakeBuilderConfig:
    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""
        self.profiling_verbosity = None


@pytest.mark.parametrize(
    ("routes", "expected"),
    [('{"peep:match_dual_gemm": {}}', "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, expected: str) -> None:
    pytest.importorskip("tensorrt")
    ctx = Pi05QuantContext(scales={"x": object()})
    config = _FakeBuilderConfig(routes)
    configure_quantized_build(config, ctx)
    assert config.build_route == expected and config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_quantized_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _gpu_trt():
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a TensorRT build needs a GPU")
    return trt


def test_quantized_projection_runs_in_bf16_without_casts(tiny) -> None:
    """Both Dequantize layers and the matmul of a quantized projection are bf16."""
    trt = _gpu_trt()
    weights, _ = tiny
    ctx = build_quant_context(weights, CFG, "fp8", precision="bf16")
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.bfloat16, (4, WIDTH))
    out = ctx.linear(network, x, f"{EXPERT_ROOT}.0.mlp.up_proj", weights)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    dequant = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequant) == 2 and all(layer.get_output(0).dtype == trt.bfloat16 for layer in dequant)
    assert not any(layer.type == trt.LayerType.CAST for layer in layers)
    assert out.dtype == trt.bfloat16


def _tiny_denoise_plan(weights, quant, width):
    """A two-layer engine with the family's RMS norm and gated GELU MLP blocks."""
    trt = _gpu_trt()
    from .. import graph, model

    builder, network, config = model._builder("bf16", False)
    configure_quantized_build(config, quant)
    x = network.add_input("x", trt.bfloat16, (8, width))
    h = x
    for layer in range(2):
        p = f"{EXPERT_ROOT}.{layer}."

        def project(t, name):
            return (
                graph.linear(network, t, weights[name + ".weight"])
                if quant is None
                else quant.linear(network, t, name, weights)
            )

        norm = graph.cast(network, graph.rms_normalise(network, h, 1e-6), trt.bfloat16)
        gate = graph.gelu_tanh(network, project(norm, p + "mlp.gate_proj"))
        hidden = graph.mul(network, gate, project(norm, p + "mlp.up_proj"))
        h = graph.add(network, h, project(hidden, p + "mlp.down_proj"))
    h.name = "y"
    network.mark_output(h)
    return model._serialize(builder, network, config, "tiny")


def test_tiny_fp8_engine_builds_and_is_smaller(tiny) -> None:
    trt = _gpu_trt()
    weights = _weights(width=256, mlp=512)
    ctx = build_quant_context(weights, CFG, "fp8", precision="bf16")
    plain = _tiny_denoise_plan(weights, None, 256)
    fp8 = _tiny_denoise_plan(weights, ctx, 256)
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8)
    assert engine is not None and engine.get_tensor_name(engine.num_io_tensors - 1) == "y"
    assert len(fp8) < len(plain)
