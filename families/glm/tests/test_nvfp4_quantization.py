# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM NVFP4 self-quantization: weight quantizer, recipe, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from .. import calibrate_nvfp4_scales, graph_ops, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    GlmQuantContext,
    _FP4Weight,
    _FP8Weight,
    calibrate_glm_nvfp4,
    configure_fp8_build,
    dequantize_nvfp4_weight,
    projection_format,
    quantize_nvfp4_weight,
)
from .test_quantization import (  # noqa: E402
    LAYERS,
    MLP,
    _activation_scales,
    _config,
    _weights,
)

E4M3_ONE = 0x38  # raw e4m3 byte of 1.0


def test_quantize_nvfp4_shapes_dtypes_and_global_scale() -> None:
    """Packed bytes are [out, in/2] uint8, block scales [out, in/16] uint8, scale_2 = amax / 2688."""
    weight = np.random.RandomState(0).randn(8, 64).astype(np.float32)
    packed, block_scale, scale_2 = quantize_nvfp4_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == (8, 32)
    assert block_scale.dtype == np.uint8 and block_scale.shape == (8, 4)
    assert scale_2 == pytest.approx(float(np.abs(weight).max()) / (6 * 448))


def test_quantize_nvfp4_global_scale_matches_the_modelopt_export_bits() -> None:
    """weight_scale_2 is amax times fp32(1 / 2688), as in ModelOpt.

    nvidia/Qwen3-8B-NVFP4 stores 0.0010811942629516125 for a weight with amax 2.90625: one bit away
    from the correctly rounded division.
    """
    weight = np.zeros((1, 16), dtype=np.float32)
    weight[0, 0] = 2.90625
    assert quantize_nvfp4_weight(weight)[2] == 0.0010811942629516125


def test_quantize_nvfp4_round_trip_error_bound_per_block() -> None:
    """Each value is within one block step (plus the saturation of the block maximum)."""
    weight = np.random.RandomState(1).randn(16, 128).astype(np.float32) * 0.03
    packed, block_scale, scale_2 = quantize_nvfp4_weight(weight)
    restored = dequantize_nvfp4_weight(packed, block_scale, scale_2).numpy()
    scale = (
        torch.from_numpy(block_scale).view(torch.float8_e4m3fn).float().numpy().repeat(16, axis=1)
    ) * scale_2
    block_amax = np.abs(weight).reshape(16, 8, 16).max(-1).repeat(16, axis=1)
    # The widest E2M1 gap is 2 (between 4 and 6): half of it is one scaled unit.
    bound = scale + np.maximum(0.0, block_amax - 6.0 * scale)
    assert (np.abs(restored - weight) <= bound * 1.0001).all()


def test_quantize_nvfp4_exact_grid_and_ties() -> None:
    """Grid values are exact; midpoints round to the even code; the low nibble holds element 0."""
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    row = np.concatenate([grid, -grid])  # 16 values, amax 6 -> block scale 448, scale_2 6/2688
    packed, block_scale, scale_2 = quantize_nvfp4_weight(row[None, :])
    assert block_scale[0, 0] == 0x7E  # e4m3 448
    restored = dequantize_nvfp4_weight(packed, block_scale, scale_2).numpy()[0]
    assert np.allclose(restored, row, rtol=1e-6, atol=0)
    # elements 0..3 are 0, 0.5, 1, 1.5 -> codes 0, 1, 2, 3, first element in the low nibble
    assert packed[0, 0] == 0x10 and packed[0, 1] == 0x32
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2, dtype=np.float32)
    packed, block_scale, scale_2 = quantize_nvfp4_weight(ties[None, :])
    restored = dequantize_nvfp4_weight(packed, block_scale, scale_2).numpy()[0]
    assert np.allclose(restored[:8], [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0], rtol=1e-6, atol=0)


def test_quantize_nvfp4_all_zero_block_gets_unit_scale() -> None:
    """An all-zero block has block scale 1 and decodes to zero, and a normal block next to it works."""
    weight = np.zeros((2, 32), dtype=np.float32)
    weight[0, 16:] = np.linspace(-1, 1, 16)
    packed, block_scale, scale_2 = quantize_nvfp4_weight(weight)
    assert block_scale[0, 0] == E4M3_ONE and block_scale[1, 0] == E4M3_ONE
    restored = dequantize_nvfp4_weight(packed, block_scale, scale_2).numpy()
    assert (restored[1] == 0).all() and (restored[0, :16] == 0).all()
    assert np.abs(restored[0, 16:] - weight[0, 16:]).max() < 0.3


def test_quantize_nvfp4_block_scale_is_clamped_not_nan() -> None:
    """A huge block ratio saturates at e4m3 448 instead of producing a NaN byte."""
    weight = np.random.RandomState(2).randn(4, 64).astype(np.float32)
    packed, block_scale, scale_2 = quantize_nvfp4_weight(weight, weight_scale_2=1e-6)
    decoded = torch.from_numpy(block_scale).view(torch.float8_e4m3fn).float()
    assert not torch.isnan(decoded).any() and float(decoded.max()) <= 448.0
    assert not np.isnan(dequantize_nvfp4_weight(packed, block_scale, 1e-6).numpy()).any()


def test_quantize_nvfp4_requires_multiple_of_16_inputs_and_a_scale() -> None:
    """in_features must be a multiple of 16; an all-zero tensor has no scale."""
    with pytest.raises(ValueError, match="multiple of 16"):
        quantize_nvfp4_weight(np.ones((4, 24), dtype=np.float32))
    with pytest.raises(ValueError, match="amax"):
        quantize_nvfp4_weight(np.zeros((4, 32), dtype=np.float32))
    assert quantization.nvfp4_supported(4096) and quantization.nvfp4_supported(13696)
    assert not quantization.nvfp4_supported(13697)


def test_recipe_formats_of_the_calibrated_shape() -> None:
    """q/k/v, the edge-layer MLP and the layer 4 down projection are FP8; the rest of the MLP is NVFP4."""
    config = ModelConfig.create_tiny(
        "glm", hidden_size=4096, intermediate_size=13696, num_hidden_layers=40
    )
    assert quantization.scale_key(config) == "glm-h4096-l40-i13696"
    fmt = lambda layer, stem: projection_format(config, layer, stem)  # noqa: E731
    assert fmt(10, "w_q") == fmt(10, "w_k") == fmt(10, "w_v") == "fp8"
    assert fmt(10, "w_gate") == fmt(10, "w_up") == fmt(10, "w_down") == "fp4"
    for layer in (0, 1, 38, 39):
        assert fmt(layer, "w_gate") == fmt(layer, "w_up") == fmt(layer, "w_down") == "fp8"
    assert fmt(4, "w_down") == "fp8" and fmt(4, "w_gate") == "fp4" and fmt(5, "w_down") == "fp4"


def test_recipe_is_refused_for_an_uncalibrated_shape() -> None:
    """A shape without a recipe fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no NVFP4 recipe"):
        projection_format(_config(), 0, "w_gate")


def _recipe_context(monkeypatch, fp8_mlp_layers=(0,), fp8_down_layers=()) -> GlmQuantContext:
    """Build the NVFP4 context of the tiny model under a test recipe."""
    monkeypatch.setitem(
        quantization.NVFP4_RECIPES,
        quantization.scale_key(_config()),
        {"fp8_mlp_layers": fp8_mlp_layers, "fp8_down_layers": fp8_down_layers},
    )
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    monkeypatch.setattr(
        quantization, "load_nvfp4_activation_scales", lambda config: _activation_scales()
    )
    return calibrate_glm_nvfp4(_config(), _weights(), graph_ops)


def test_context_mixes_nvfp4_mlp_and_fp8(monkeypatch) -> None:
    """q/k/v and the FP8 layers use FP8 weights; the other MLP uses NVFP4 bytes of the loaded weights."""
    ctx = _recipe_context(monkeypatch, fp8_mlp_layers=(0,))
    assert isinstance(ctx.scales["layer.0.w_q"].weight, _FP8Weight)
    assert isinstance(ctx.scales["layer.0.w_gate"].weight, _FP8Weight)
    entry = ctx.scales["layer.1.w_down"].weight
    assert isinstance(entry, _FP4Weight)
    assert (entry.out_features, entry.in_features) == (512, MLP)
    source = np.asarray(_weights()["layer.1.w_down"], dtype=np.float32).T
    packed, block_scale, scale_2 = quantize_nvfp4_weight(source)
    assert np.array_equal(entry.packed, packed) and np.array_equal(entry.block_scale, block_scale)
    assert entry.weight_scale_2 == pytest.approx(scale_2)
    assert not ctx.should_quantize("layer.0.w_o")
    assert ctx.has_fp4_weights


def test_down_projection_can_stay_fp8(monkeypatch) -> None:
    """A layer listed in fp8_down_layers has an FP8 down projection and NVFP4 gate and up."""
    ctx = _recipe_context(monkeypatch, fp8_mlp_layers=(), fp8_down_layers=(1,))
    assert isinstance(ctx.scales["layer.1.w_down"].weight, _FP8Weight)
    assert isinstance(ctx.scales["layer.1.w_up"].weight, _FP4Weight)
    assert isinstance(ctx.scales["layer.0.w_down"].weight, _FP4Weight)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    all_build_routes = '{"peep:match_dual_gemm": {}}'

    def __init__(self) -> None:
        self.build_route = ""
        self.profiling_verbosity = None


def test_nvfp4_build_keeps_the_dual_gemm_kernel(monkeypatch) -> None:
    """NVFP4 builds keep the fused gate/up kernel; FP8-only builds turn it off."""
    nvfp4 = _FakeBuilderConfig()
    configure_fp8_build(nvfp4, _recipe_context(monkeypatch))
    assert nvfp4.build_route == "" and nvfp4.profiling_verbosity is not None
    fp8 = _FakeBuilderConfig()
    configure_fp8_build(fp8, GlmQuantContext(scales={}, graph_ops=graph_ops))
    assert fp8.build_route == "-peep:match_dual_gemm=off"


def test_projection_without_nvfp4_scale_is_rejected(monkeypatch) -> None:
    """A projection with no calibrated scale fails loudly and names the calibration tool."""
    monkeypatch.setitem(
        quantization.NVFP4_RECIPES,
        quantization.scale_key(_config()),
        {"fp8_mlp_layers": (), "fp8_down_layers": ()},
    )
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    monkeypatch.setattr(
        quantization,
        "load_nvfp4_activation_scales",
        lambda config: _activation_scales(("w_down",)),
    )
    with pytest.raises(
        ValueError, match=r"layer\.0\.w_down has no activation scale.*calibrate_nvfp4_scales"
    ):
        calibrate_glm_nvfp4(_config(), _weights(), graph_ops)


def test_projection_without_fp8_scale_in_the_nvfp4_build_is_rejected(monkeypatch) -> None:
    """The FP8 projections of the NVFP4 recipe (q/k/v) are checked the same way."""
    monkeypatch.setitem(
        quantization.NVFP4_RECIPES,
        quantization.scale_key(_config()),
        {"fp8_mlp_layers": (), "fp8_down_layers": ()},
    )
    fp8_scales = _activation_scales()
    fp8_scales.pop(next(iter(fp8_scales)))
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: fp8_scales)
    monkeypatch.setattr(
        quantization, "load_nvfp4_activation_scales", lambda config: _activation_scales()
    )
    with pytest.raises(ValueError, match="fp8_activation_scales.json"):
        calibrate_glm_nvfp4(_config(), _weights(), graph_ops)


def test_missing_nvfp4_scale_entry_is_refused() -> None:
    """A model shape without committed NVFP4 scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated NVFP4 activation scales"):
        quantization.load_nvfp4_activation_scales(_config())


def test_committed_nvfp4_scales_cover_every_quantized_projection() -> None:
    """Every committed shape has a positive scale for every projection."""
    path = Path(quantization.__file__).parent / quantization.NVFP4_ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {
            f"layer.{layer}.{stem}" for layer in layers for stem in quantization.WEIGHT_STEMS
        }
        assert set(scales) == expected, key


def test_calibration_tool_samples_both_prompt_formats() -> None:
    """The calibration tool accepts chat messages and plain text rows."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_nvfp4_scales.sample_text(
        _NoTemplate(), {"messages": [{"role": "user", "content": "a"}], "max_tokens": 7}
    )
    assert text == ("a", 7)
    assert calibrate_nvfp4_scales.sample_text(_NoTemplate(), {"text": "b"}) == ("b", 512)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_nvfp4_weights(monkeypatch) -> None:
    """The NVFP4 engine builds and stores 4-bit (not full-precision) MLP weights."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 needs a Blackwell GPU")
    ctx = _recipe_context(monkeypatch)
    glm_rope = {"partial_rotary_factor": 0.5, "interleaved_rope": True}
    plain = build_standard_decoder_engine(_config(), _weights(), 4, precision="fp16", **glm_rope)
    quantized = build_standard_decoder_engine(
        _config(), _weights(), 4, precision="fp16", quant_ctx=ctx, **glm_rope
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    assert LAYERS == 2 and len(quantized) < len(plain)


def test_calibration_tool_cleans_conversations() -> None:
    """Conversations start with a user turn and need at least two turns."""
    turns = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": ""},
    ]
    cleaned = calibrate_nvfp4_scales._clean(turns)
    assert [t["role"] for t in cleaned] == ["user", "assistant"]
    assert calibrate_nvfp4_scales._clean(turns[:3]) is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_nvfp4_scales, "fetch_mix", lambda d, n, o: seen.update(fetch=(d, n, o))
    )
    monkeypatch.setattr(
        calibrate_nvfp4_scales, "calibrate", lambda d, p, n: seen.update(calibrate=(d, p, n))
    )
    monkeypatch.setattr(
        "sys.argv",
        ["x", "fetch", "--model-dir", "m", "--nemotron", "n.jsonl", "--out", str(tmp_path / "o")],
    )
    calibrate_nvfp4_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_nvfp4_scales.main()
    assert seen["fetch"] == (Path("m"), Path("n.jsonl"), tmp_path / "o")
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512)
