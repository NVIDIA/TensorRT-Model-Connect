# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.personaplex import calibrate_fp8_scales, graph_ops, model  # noqa: E402
from families.personaplex import quantization as quantization_module  # noqa: E402
from families.personaplex.checkpoint_mapper import _open_safetensors  # noqa: E402
from families.personaplex.config import ModelConfig  # noqa: E402
from families.personaplex.default_decoder import build_standard_decoder_engine  # noqa: E402
from families.personaplex.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    PROJECTION_STEMS,
    PersonaPlexQuantContext,
    configure_fp8_build,
    create_quant_context,
    quantize_fp8_weight,
    scale_key,
)

# Temporal: hidden H with 32 heads (the loader fixes the head count), 2 layers. Depth: hidden DH
# with 16 heads, K codebooks.
H, I, LAYERS, TV, AV = 256, 512, 2, 40, 20
DH, DI, DL, K = 32, 64, 2, 3
TEMPORAL_PROJECTIONS = LAYERS * len(PROJECTION_STEMS)


def _write_checkpoint(path: Path) -> None:
    """Write a random tiny PersonaPlex (Moshi layout) checkpoint."""
    rng = np.random.default_rng(0)

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    ones = lambda *shape: np.ones(shape, np.float32)  # noqa: E731
    t = {
        "text_emb.weight": w(TV, H),
        "text_linear.weight": w(TV - 1, H),
        "out_norm.alpha": ones(1, 1, H),
        "depformer_text_emb.weight": w(TV, DH),
    }
    for i in range(LAYERS):
        p = f"transformer.layers.{i}"
        t[f"{p}.norm1.alpha"] = ones(1, 1, H)
        t[f"{p}.norm2.alpha"] = ones(1, 1, H)
        t[f"{p}.self_attn.in_proj_weight"] = w(3 * H, H)
        t[f"{p}.self_attn.out_proj.weight"] = w(H, H)
        t[f"{p}.gating.linear_in.weight"] = w(2 * I, H)
        t[f"{p}.gating.linear_out.weight"] = w(H, I)
    for i in range(DL):
        p = f"depformer.layers.{i}"
        t[f"{p}.norm1.alpha"] = ones(1, 1, DH)
        t[f"{p}.norm2.alpha"] = ones(1, 1, DH)
        t[f"{p}.self_attn.in_proj_weight"] = w(K * 3 * DH, DH)
        t[f"{p}.self_attn.out_proj.weight"] = w(K * DH, DH)
        for k in range(K):
            t[f"{p}.gating.{k}.linear_in.weight"] = w(2 * DI, DH)
            t[f"{p}.gating.{k}.linear_out.weight"] = w(DH, DI)
    for k in range(K):
        t[f"emb.{k}.weight"] = w(AV, H)
        t[f"linears.{k}.weight"] = w(AV - 1, DH)
        t[f"depformer_in.{k}.weight"] = w(DH, H)
        if k < K - 1:
            t[f"depformer_emb.{k}.weight"] = w(AV, DH)
    safetensors_numpy.save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "personaplex"}), encoding="utf-8")


def _scales(value: float = 0.02) -> dict[str, float]:
    return {f"layer.{i}.{s}": value for i in range(LAYERS) for s in PROJECTION_STEMS}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny checkpoint plus an activation-scale table that covers every temporal projection."""
    _write_checkpoint(tmp_path)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda inferred: _scales())
    return tmp_path


def _load(path: Path, quantization: str | None):
    config = ModelConfig.from_dir(path)
    return model._PersonaPlexModel().load_weights(str(path), config, quantization=quantization)


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


def test_every_temporal_projection_is_quantized_except_output_projections(tiny) -> None:
    """q, k, v, gate and up get FP8 weights; both output projections, depth and lm head do not."""
    weights = _load(tiny, "fp8")
    ctx = weights["_quant_ctx"]
    assert len(ctx.scales) == TEMPORAL_PROJECTIONS
    assert ctx.should_quantize("layer.1.w_up") and not ctx.should_quantize("layer.0.w_o")
    assert not ctx.should_quantize("layer.1.w_down")
    entry = ctx.scales["layer.0.w_gate"]
    assert (entry.weight.out_features, entry.weight.in_features) == (I, H)
    assert entry.weight.packed.dtype == np.uint8
    # quantized projections keep a shape-only placeholder, the rest is a real [in, out] matrix
    placeholder = weights["temporal.layer.0.w_q"]
    assert placeholder.shape == (H, H) and placeholder.strides == (0, 0)
    assert weights["temporal.layer.0.w_o"].strides != (0, 0)
    assert weights["temporal.layer.0.w_down"].strides != (0, 0)
    assert weights["temporal.w_out"].shape == (H, TV - 1)
    assert weights["depth_cb0.layer.0.w_q"].strides != (0, 0)


def test_unquantized_load_has_no_context(tiny) -> None:
    """Without quantization nothing is replaced."""
    weights = _load(tiny, None)
    assert "_quant_ctx" not in weights
    assert weights["temporal.layer.0.w_q"].strides != (0, 0)


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    _write_checkpoint(tmp_path)
    scales = _scales()
    del scales["layer.0.w_up"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda inferred: scales)
    weights = _load(tmp_path, "fp8")
    ctx = weights["_quant_ctx"]
    assert not ctx.should_quantize("layer.0.w_up") and ctx.should_quantize("layer.0.w_gate")
    assert weights["temporal.layer.0.w_up"].strides != (0, 0)


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    tensors = safetensors_numpy.load_file(str(tiny / "model.safetensors"))
    tensors["transformer.layers.0.gating.linear_in.weight_scale"] = np.ones((1,), np.float32)
    safetensors_numpy.save_file(tensors, str(tiny / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        _load(tiny, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    readers = _open_safetensors(tiny)
    with pytest.raises(NotImplementedError, match="nvfp4"):
        create_quant_context(readers, {}, graph_ops, "nvfp4")


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A model shape with no committed scales raises and names the calibration tool."""
    _write_checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        _load(tmp_path, "fp8")


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a positive scale for each projection of every layer."""
    path = Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table, "no committed activation scales"
    for key, scales in table.items():
        match = re.fullmatch(r"personaplex-h(\d+)-l(\d+)-i(\d+)", key)
        assert match, key
        hidden, layers, inter = (int(g) for g in match.groups())
        inferred = {"hidden_size": hidden, "num_hidden_layers": layers, "intermediate_size": inter}
        assert scale_key(inferred) == key
        assert set(scales) == {f"layer.{i}.{s}" for i in range(layers) for s in PROJECTION_STEMS}
        assert all(value > 0 for value in scales.values()), key


def test_calibration_tool_cli_parses_command(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts calibrate and passes the options on."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda *args: seen.update(calibrate=args)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "calibrate",
            "--model-dir",
            str(tmp_path),
            "--librispeech-dir",
            "ls",
            "--count",
            "3",
            "--official-repo",
            "src",
        ],
    )
    calibrate_fp8_scales.main()
    assert seen["calibrate"] == (tmp_path, Path("ls"), 16.0, 3, Path("src"))


def test_calibration_speech_is_continuous_24khz(tmp_path: Path) -> None:
    """The calibration speech streams have the requested length and no silent gaps beyond the joins."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    soundfile = pytest.importorskip("soundfile")
    import io

    rng = np.random.default_rng(2)
    rows = []
    for _ in range(4):
        buffer = io.BytesIO()
        soundfile.write(
            buffer, rng.standard_normal(16000).astype(np.float32) * 0.1, 16000, format="WAV"
        )
        rows.append({"audio": {"bytes": buffer.getvalue()}})
    for split in calibrate_fp8_scales._SPLITS:
        (tmp_path / split).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / split)
    streams = calibrate_fp8_scales.load_speech(tmp_path, seconds=3.0, count=2)
    assert len(streams) == 2
    assert all(s.shape == (1, 72000) and s.dtype == np.float32 for s in streams)
    assert float(np.abs(streams[0]).max()) > 0


def test_quantized_decoder_engine_builds(tiny) -> None:
    """A temporal-style decoder builds with FP8 projections and is smaller than fp16."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    plain = _load(tiny, None)
    quantized = _load(tiny, "fp8")
    ctx = quantized["_quant_ctx"]
    config = ModelConfig(
        model_type="personaplex",
        vocab_size=TV,
        hidden_size=H,
        num_hidden_layers=LAYERS,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=I,
        rms_norm_eps=1e-8,
        _head_dim=H // 32,
    )

    def engine(weights, quant_ctx):
        decoder = {
            k[len("temporal.") :]: v for k, v in weights.items() if k.startswith("temporal.")
        }
        decoder["_kv_attention_size"] = H
        return build_standard_decoder_engine(
            config,
            decoder,
            16,
            precision="fp16",
            quant_ctx=quant_ctx,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            interleaved_rope=True,
            embed_input=True,
            hidden_state_output=True,
        )

    fp16_plan = engine(plain, None)
    fp8_plan = engine(quantized, ctx)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    built = runtime.deserialize_cuda_engine(fp8_plan)
    names = {built.get_tensor_name(i) for i in range(built.num_io_tensors)}
    assert {"logits", "hidden_state", "input_embed"} <= names
    assert len(fp8_plan) < len(fp16_plan)


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
    ctx = PersonaPlexQuantContext(activation_scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None
