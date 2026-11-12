# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_ops, quantization  # noqa: E402
from .. import model as model_module  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    LanceQuantContext,
    calibrate_lance_fp8,
    configure_fp8_build,
    load_activation_scales,
    quantize_fp8_weight,
    scale_key,
)

HIDDEN, MLP, VOCAB, LAYERS, HEADS, KV_HEADS = 512, 1024, 32, 2, 4, 2
HEAD_DIM = HIDDEN // HEADS
KV = KV_HEADS * HEAD_DIM

_SHAPES = {
    "w_q": (HIDDEN, HIDDEN),
    "w_k": (HIDDEN, KV),
    "w_v": (HIDDEN, KV),
    "w_o": (HIDDEN, HIDDEN),
    "w_gate": (HIDDEN, MLP),
    "w_up": (HIDDEN, MLP),
    "w_down": (MLP, HIDDEN),
}
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)
_QUANTIZED = tuple(_WEIGHT_STEMS.values())


def _config() -> ModelConfig:
    """Build a small Lance-shaped model config for the tests."""
    return ModelConfig.create_tiny(
        "lance",
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        intermediate_size=MLP,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    rng = np.random.RandomState(0)
    weights = WeightDict()
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float16)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = rng.randn(*shape).astype(np.float16)
        weights[f"{prefix}.q_bias"] = np.zeros(HIDDEN, dtype=np.float16)
        weights[f"{prefix}.k_bias"] = np.zeros(KV, dtype=np.float16)
        weights[f"{prefix}.v_bias"] = np.zeros(KV, dtype=np.float16)
        weights[f"{prefix}.q_norm"] = np.ones(HIDDEN, dtype=np.float32)
        weights[f"{prefix}.k_norm"] = np.ones(KV, dtype=np.float32)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_out"] = rng.randn(HIDDEN, VOCAB).astype(np.float16)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = KV
    weights["_mlp_size"] = MLP
    return weights


def _table(excluded: tuple[str, ...] = (), skip: tuple[str, ...] = ()):
    """Return ``(input_scales, excluded)`` as ``load_activation_scales`` would."""
    scales = {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(LAYERS)
        for stem in _QUANTIZED
        if f"layer.{layer}.{stem}" not in skip
    }
    for name in excluded:
        scales.pop(name, None)
    return scales, frozenset(excluded)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)

    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_scale_key_identifies_model() -> None:
    """The scale key names the model type and shape."""
    assert scale_key(_config()) == f"lance-h{HIDDEN}-l{LAYERS}-i{MLP}"


def test_unknown_model_shape_is_refused() -> None:
    """A model config without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config())


def test_committed_scales_cover_every_projection_or_exclude_it() -> None:
    """Every committed shape has a positive scale or an explicit exclusion per projection."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert "lance-h2048-l36-i11008" in table
    for key, entry in table.items():
        scales, excluded = entry["input_scales"], set(entry["excluded"])
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _QUANTIZED}
        assert set(scales) == expected, key
        assert excluded <= expected, key


def test_dequantize_runs_in_fp32() -> None:
    """A bf16 Dequantize returns wrong logits on TensorRT 11.4, so fp32 is the default."""
    assert quantization.DEQUANTIZE_DTYPE == "float32"


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in _QUANTIZED


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _table())
    ctx = calibrate_lance_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_QUANTIZED)
    entry = ctx.scales["layer.1.w_down"]
    in_f, out_f = _SHAPES["w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = np.asarray(weights["layer.1.w_down"], dtype=np.float32).T
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.0.w_o")


def test_excluded_projection_stays_unquantized(monkeypatch) -> None:
    """A projection listed as excluded is left unquantized, the others are quantized."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _table(("layer.1.w_down",))
    )
    ctx = calibrate_lance_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.1.w_down")
    assert ctx.should_quantize("layer.0.w_down")
    assert "layer.1.w_down" in ctx.excluded


def test_projection_without_a_scale_or_exclusion_raises(monkeypatch) -> None:
    """A projection with no scale entry fails the build and names the projection and the tool."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _table(skip=("layer.0.w_up",))
    )
    with pytest.raises(ValueError, match=r"layer\.0\.w_up.*calibrate_fp8_scales\.py"):
        calibrate_lance_fp8(_config(), _weights(), graph_ops)


def test_context_refuses_an_unlisted_projection_at_graph_time() -> None:
    """The graph helper also refuses a quantizable projection that is neither scaled nor excluded."""
    ctx = LanceQuantContext(scales={}, graph_ops=graph_ops)
    with pytest.raises(ValueError, match=r"layer\.3\.w_gate.*calibrate_fp8_scales\.py"):
        ctx.maybe_quantized_matmul(None, None, 4, 4, np.zeros((4, 4)), "layer.3.w_gate")


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _table())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lance_fp8(_config(), weights, graph_ops)


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
    ctx = LanceQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def test_build_refuses_other_quantization_modes(tmp_path: Path) -> None:
    """Only fp8 is accepted besides none."""
    request = SimpleNamespace(
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=1,
        fp32_layers=None,
        quantization="nvfp4",
        model_dir=str(tmp_path),
    )
    with pytest.raises(NotImplementedError, match="does not support quantization"):
        model_module.build(request, None)


def test_build_refuses_fp16_fp8(tmp_path: Path) -> None:
    """The FP8 build needs bf16 (fp16 cannot hold the down-projection inputs)."""
    (tmp_path / "Lance_3B").mkdir()
    (tmp_path / "Qwen2.5-VL-ViT").mkdir()
    for path in (
        tmp_path / "Lance_3B" / "llm_config.json",
        tmp_path / "Lance_3B" / "model.safetensors",
        tmp_path / "Qwen2.5-VL-ViT" / "vit.safetensors",
    ):
        path.write_text(
            json.dumps(
                {
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "intermediate_size": 32,
                    "vocab_size": 8,
                }
            )
        )
    request = SimpleNamespace(
        mtp_seq_len=None,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=1,
        fp32_layers=None,
        quantization="fp8",
        precision="fp16",
        max_sequence_length=None,
        model_dir=str(tmp_path),
    )
    with pytest.raises(NotImplementedError, match="requires precision bf16"):
        model_module.build(request, None)


def _build_tiny(role: str, quant_ctx) -> bytes:
    config = _config()
    config.raw["_decoder_engine_role"] = role
    return build_standard_decoder_engine(
        config, _weights(), 16, precision="fp16", quant_ctx=quant_ctx, embed_input=True
    )


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_engine_builds_with_fp8_weights(monkeypatch, role: str) -> None:
    """Both decoder engines build quantized and store FP8 (not full-precision) weights."""
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _table())
    ctx = calibrate_lance_fp8(_config(), _weights(), graph_ops)

    plain = _build_tiny(role, None)
    quantized = _build_tiny(role, ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 6 of 7 projections x 2 layers: FP8 halves the fp16 projection bytes.
    assert len(quantized) < len(plain)


def test_calibration_tool_scales_and_outliers() -> None:
    """Scales are amax / 448 and an input far above the median of its kind is excluded."""
    amax = {f"layer.{i}.{stem}": 4.48 for i in range(10) for stem in _QUANTIZED}
    amax["layer.3.w_down"] = 4.48 * 50
    scales, excluded = calibrate_fp8_scales.scales_from_amax(amax, 10)
    assert scales["layer.0.w_q"] == pytest.approx(0.01)
    assert excluded == ["layer.3.w_down"]


def test_calibration_tool_refuses_a_projection_without_data() -> None:
    """A projection the calibration never saw fails instead of getting a default scale."""
    amax = {f"layer.{i}.{stem}": 1.0 for i in range(2) for stem in _QUANTIZED}
    del amax["layer.1.w_k"]
    with pytest.raises(RuntimeError, match=r"layer\.1\.w_k"):
        calibrate_fp8_scales.scales_from_amax(amax, 2)


def test_calibration_tool_mrope_positions_follow_the_native_runtime() -> None:
    """Image tokens get (t, h, w) positions; text after the image continues from base + grid."""
    token = calibrate_fp8_scales.IMAGE_TOKEN_ID
    grid = calibrate_fp8_scales.GRID
    ids = [1, 2] + [token] * (grid * grid) + [3, 4]
    positions = calibrate_fp8_scales.mrope_positions(ids)
    assert positions[0] == (0, 0, 0) and positions[1] == (1, 1, 1)
    assert positions[2] == (2, 2, 2)
    assert positions[2 + grid + 1] == (2, 3, 3)
    assert positions[-2] == (2 + grid, 2 + grid, 2 + grid)
    assert positions[-1] == (3 + grid, 3 + grid, 3 + grid)
    assert calibrate_fp8_scales.mrope_positions([1, 2, 3]) == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


def test_calibration_tool_cli_parses_calibrate(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts the calibrate command."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "calibrate",
            "--model-dir",
            str(tmp_path),
            "--image-parquet",
            "a.parquet",
            "--text-prompts",
            "p.jsonl",
        ],
    )
    calibrate_fp8_scales.main()
    assert seen["image_samples"] == 320 and seen["text_samples"] == 192
