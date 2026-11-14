# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lerobot_groot FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import action_head, calibrate_fp8_scales, graph, quantization  # noqa: E402
from ..quantization import (  # noqa: E402
    BACKBONE_ROOT,
    DIT_ROOT,
    calibrate_groot,
    configure_fp8_build,
    load_activation_scales,
    projection_kind,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

LAYERS = 16
COUNT = action_head.LAYERS * 6


def _names() -> list[str]:
    return projection_names(action_head.LAYERS)


def _weights(seed: int = 0, size: int = 8) -> dict[str, np.ndarray]:
    """Small stand-in weights under every quantized projection name and the first backbone query."""
    rng = np.random.default_rng(seed)
    names = _names() + [f"{BACKBONE_ROOT}.0.self_attn.q_proj"]
    return {f"{name}.weight": (rng.standard_normal((size, size)) * 0.05).astype(np.float32) for name in names}


def _context(weights, monkeypatch, table=None, **overrides):
    stored = {name: 0.01 for name in _names()} if table is None else table
    monkeypatch.setattr(quantization, "load_activation_scales", lambda key: stored)
    values = {"backbone_layers": LAYERS, "dit_layers": action_head.LAYERS,
              "embodiment_id": 0, "quantization": "fp8", "precision": "bf16"}
    values.update(overrides)
    return calibrate_groot(weights, **values)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    weight = np.array([[0.5, -1.0], [0.25, 0.0]], dtype=np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(1.0 / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == (2, 2)
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    np.testing.assert_allclose(restored, weight, rtol=0.07)


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    weight = np.linspace(-3.0, 3.0, 4096, dtype=np.float32).reshape(64, 64)
    packed, _ = quantize_fp8_weight(weight)
    assert np.isfinite(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).all()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_projection_names_cover_every_dit_block() -> None:
    names = _names()
    assert len(names) == len(set(names)) == COUNT
    assert f"{DIT_ROOT}.0.attn1.to_k" in names and f"{DIT_ROOT}.31.ff.net.2" in names
    assert f"{DIT_ROOT}.17.attn1.to_out.0" in names
    # Backbone, VL blocks, modulation, timestep embedders and the DiT output stay unquantized.
    assert all(n.startswith(DIT_ROOT + ".") for n in names)
    assert not any("norm1.linear" in n or "timestep" in n or "proj_out" in n for n in names)


def test_projection_kind_separates_cross_and_self_attention_blocks() -> None:
    cross = action_head.CROSS_BLOCKS
    assert projection_kind(f"{DIT_ROOT}.0.attn1.to_k", cross) == "attn1.to_k.cross"
    assert projection_kind(f"{DIT_ROOT}.1.attn1.to_k", cross) == "attn1.to_k.self"
    assert projection_kind(f"{DIT_ROOT}.2.ff.net.2", cross) == "ff.net.2.cross"
    for name in (f"{BACKBONE_ROOT}.7.mlp.up_proj", f"{DIT_ROOT}.0.norm1.linear"):
        with pytest.raises(ValueError, match="not a quantized"):
            projection_kind(name, cross)


def test_scale_key_names_the_layers_embodiment_and_weights() -> None:
    weights = _weights()
    key = scale_key(weights, backbone_layers=LAYERS, embodiment_id=3)
    assert key.startswith("groot-n17-l16-e3-w")
    other = _weights(seed=1)
    assert scale_key(other, backbone_layers=LAYERS, embodiment_id=3) != key


def test_unknown_checkpoint_is_refused_and_names_the_tool() -> None:
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        load_activation_scales("groot-n17-l16-e0-w1.0000e+00-1.0000e+00")


def test_committed_scales_cover_every_projection() -> None:
    table = json.loads((Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME).read_text())
    assert table
    names = set(_names())
    for key, scales in table.items():
        assert key.startswith("groot-n17-l16-")
        assert set(scales) == names, key
        values = [v for v in scales.values() if v is not None]
        assert values and all(np.isfinite(v) and v > 0 for v in values)


def test_fp8_e2e_manifest_mirrors_the_bf16_cases_and_has_thresholds() -> None:
    tests = Path(__file__).parent
    read = lambda name: json.loads((tests / "manifests" / f"{name}.json").read_text())  # noqa: E731
    fp8, bf16 = read("groot-n17-libero-spatial-fp8"), read("groot-n17-libero-spatial-bf16")
    assert fp8["quantization"] == "fp8" and fp8["precision"] == "bf16"
    assert (fp8["hf_id"], fp8["hf_revision"]) == (bf16["hf_id"], bf16["hf_revision"])
    assert [case["inputs"] for case in fp8["testcases"]] == [case["inputs"] for case in bf16["testcases"]]
    assert any(case["premerge"] for case in fp8["testcases"])
    for case in fp8["testcases"]:
        limits = json.loads((tests / "thresholds" / f"{case['name']}.json").read_text())["threshold_overrides"]
        assert {"max_abs_vs_fp32", "mean_abs_vs_fp32", "cosine_vs_fp32_min", "max_abs_vs_bf16"} <= set(limits)


def test_context_quantizes_every_projection(monkeypatch) -> None:
    weights = _weights()
    ctx = _context(weights, monkeypatch)
    assert len(ctx.projections) == COUNT
    name = f"{DIT_ROOT}.2.ff.net.0.proj"
    projection = ctx.get(name)
    assert projection.input_scale == 0.01 and projection.packed.shape == (8, 8)
    assert projection.weight_scale == pytest.approx(np.abs(weights[f"{name}.weight"]).max() / 448.0)
    assert ctx.get("action_head.model.proj_out_2") is None and ctx.get(f"{BACKBONE_ROOT}.0.self_attn.q_proj") is None
    assert quantization.fp8(None, name) is None and quantization.fp8(ctx, name) is projection


def test_null_entry_stays_unquantized(monkeypatch) -> None:
    weights = _weights()
    table = {name: 0.01 for name in _names()}
    table[f"{DIT_ROOT}.1.ff.net.2"] = None
    ctx = _context(weights, monkeypatch, table)
    assert ctx.get(f"{DIT_ROOT}.1.ff.net.2") is None and len(ctx.projections) == COUNT - 1


def test_projection_without_a_scale_is_refused(monkeypatch) -> None:
    table = {name: 0.01 for name in _names()}
    del table[f"{DIT_ROOT}.5.ff.net.2"]
    with pytest.raises(ValueError, match="has no activation scale"):
        _context(_weights(), monkeypatch, table)


def test_scale_for_an_unknown_projection_is_refused(monkeypatch) -> None:
    table = {name: 0.01 for name in _names()}
    table[f"{BACKBONE_ROOT}.2.mlp.up_proj"] = 0.01
    with pytest.raises(ValueError, match="does not quantize"):
        _context(_weights(), monkeypatch, table)


def test_bad_scale_value_is_refused(monkeypatch) -> None:
    table = {name: 0.01 for name in _names()}
    table[f"{DIT_ROOT}.0.attn1.to_q"] = -1.0
    with pytest.raises(ValueError, match="finite and positive"):
        _context(_weights(), monkeypatch, table)


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    weights = _weights()
    weights[f"{DIT_ROOT}.0.ff.net.2.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        _context(weights, monkeypatch)


@pytest.mark.parametrize(("overrides", "error", "message"), [
    ({"precision": "fp32"}, ValueError, "needs precision bf16"),
    ({"quantization": "nvfp4"}, NotImplementedError, "supports quantization"),
])
def test_unsupported_settings_are_refused(monkeypatch, overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _context(_weights(), monkeypatch, **overrides)


def test_fp8_build_options_only_for_quantized_builds() -> None:
    config = SimpleNamespace(profiling_verbosity=None, all_build_routes="-peep:match_dual_gemm=on", build_route="")
    configure_fp8_build(config, None)
    assert config.profiling_verbosity is None and config.build_route == ""
    configure_fp8_build(config, object())
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    assert config.build_route == "-peep:match_dual_gemm=off"
    plain = SimpleNamespace(profiling_verbosity=None, all_build_routes="", build_route="")
    configure_fp8_build(plain, object())
    assert plain.build_route == ""


def test_projection_name_matches_only_quantized_modules() -> None:
    match = calibrate_fp8_scales.projection_name
    assert match(f"{DIT_ROOT}.30.attn1.to_out.0") is not None
    assert match(f"{DIT_ROOT}.1.ff.net.0.proj") is not None
    for name in (f"{BACKBONE_ROOT}.3.self_attn.q_proj", f"{DIT_ROOT}.0.norm1.linear", f"{DIT_ROOT}.0.attn1",
                 "action_head.vl_self_attention.transformer_blocks.1.ff.net.0.proj", f"{DIT_ROOT}.3.attn1.to_q.x"):
        assert match(name) is None, name


def test_outlier_projections_become_null() -> None:
    cross = action_head.CROSS_BLOCKS
    scales = {f"{DIT_ROOT}.{i}.ff.net.2": 0.01 for i in range(action_head.LAYERS)}
    scales[f"{DIT_ROOT}.4.ff.net.2"] = 0.5
    scales.update({f"{DIT_ROOT}.{i}.attn1.to_k": (1.0 if i in cross else 0.01) for i in range(8)})
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales, cross)
    # The cross-attention keys read other tokens: compared among themselves, they are no outliers.
    assert excluded == [f"{DIT_ROOT}.4.ff.net.2"] and table[excluded[0]] is None
    assert table[f"{DIT_ROOT}.0.attn1.to_k"] == 1.0


def test_episode_split_is_disjoint_and_skips_the_fixtures() -> None:
    episodes = list(range(1000, 1500))
    calibration, held_out = calibrate_fp8_scales.split_episodes(episodes, 32, 50)
    assert len(calibration) == 32 and len(held_out) == 50
    assert not set(calibration) & set(held_out)
    assert not set(calibration + held_out) & set(calibrate_fp8_scales.FIXTURE_EPISODES)
    assert calibrate_fp8_scales.frame_positions(100, 4) == [12, 37, 62, 87]


def test_calibration_cli_has_fetch_and_calibrate(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_frames", lambda *a: calls.append(("fetch", a)))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *a: calls.append(("calibrate", a)))
    calibrate_fp8_scales.main(["fetch", "--out", str(tmp_path)])
    calibrate_fp8_scales.main(["calibrate", "--model-dir", "m", "--base-dir", "b", "--tokenizer-dir", "t",
                               "--frames", str(tmp_path)])
    assert calls[0] == ("fetch", (tmp_path, 32, 50, 4))
    assert calls[1][0] == "calibrate" and calls[1][1][3] == tmp_path


# Wide enough that the FP8 weight savings dominate the plan overhead.
WIDTH, HEADS, TOKENS = 512, 4, 16


def _tiny_block_plan(quant):
    """A pre-norm attention + MLP block through ``graph.linear``: FP8 when ``quant`` is given."""
    rng = np.random.default_rng(0)
    names = ("q", "k", "v", "o", "up", "down")
    weights = {n: (rng.standard_normal((WIDTH, WIDTH)) * 0.05).astype(np.float32) for n in names}
    ctx = None
    if quant:
        keep: list = []
        projections = {}
        for n in names:
            packed, scale = quantize_fp8_weight(weights[n])
            projections[n] = quantization.FP8Projection(packed, scale, 4.0 / 448.0, keep)
        ctx = quantization.GrootQuantContext(projections)
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    configure_fp8_build(config, ctx)
    x = network.add_input("x", trt.float32, (TOKENS, WIDTH))
    dtype = trt.bfloat16
    norm = graph.layer_norm(network, x, 1e-5)
    q, k, v = (graph.linear(network, norm, weights[n], dtype=dtype, fp8=quantization.fp8(ctx, n)) for n in "qkv")
    context = graph.attention(network, q, k, v, heads=HEADS, kv_heads=HEADS, head_dim=WIDTH // HEADS, dtype=dtype)
    h = graph.add(network, x, graph.f32(network, graph.linear(network, context, weights["o"], dtype=dtype,
                                                              fp8=quantization.fp8(ctx, "o"))))
    hidden = graph.gelu_tanh(network, graph.linear(network, graph.layer_norm(network, h, 1e-5), weights["up"],
                                                   dtype=dtype, fp8=quantization.fp8(ctx, "up")))
    out = graph.add(network, h, graph.f32(network, graph.linear(network, hidden, weights["down"], dtype=dtype,
                                                                fp8=quantization.fp8(ctx, "down"))))
    graph.mark_output(network, out, "y")
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    return bytes(plan)


def _run(plan: bytes, x: np.ndarray) -> np.ndarray:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    source = torch.from_numpy(x).cuda()
    result = torch.empty((TOKENS, WIDTH), dtype=torch.float32, device="cuda")
    context.set_tensor_address("x", source.data_ptr())
    context.set_tensor_address("y", result.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return result.cpu().numpy()


def test_engine_builds_with_fp8_weights_and_is_smaller() -> None:
    plain, quantized = _tiny_block_plan(False), _tiny_block_plan(True)
    # Six 512x512 projections: 3 MiB in bfloat16, 1.5 MiB in FP8.
    assert len(quantized) < len(plain) - 1_000_000
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(quantized)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    assert "FP8" in info or "Float8" in info or "e4m3" in info
    x = (np.random.default_rng(1).standard_normal((TOKENS, WIDTH))).astype(np.float32)
    want, got = _run(plain, x), _run(quantized, x)
    assert np.isfinite(got).all()
    cosine = float((want * got).sum() / (np.linalg.norm(want) * np.linalg.norm(got)))
    assert cosine > 0.99, cosine
