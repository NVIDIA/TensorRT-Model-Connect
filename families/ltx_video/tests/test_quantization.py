# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video FP8 self-quantization: weight scales, the scale table, the calibration tool, the build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales, model
from .. import ltx_dit_builder as dit
from .. import quantization as quantization_module
from ..checkpoint_mapper import WeightDict
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    QUANTIZED,
    LTXQuantContext,
    calibrate_ltx,
    configure_build,
    is_quantized_checkpoint,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
)

DIM, HEADS, FFN, LAYERS = 128, 4, 256, 2
CHANNELS, TEXT_DIM, TEXT_LEN = 16, 64, 8
SHAPE = {"dim": DIM, "num_layers": LAYERS, "ffn_dim": FFN}


def _weights(seed: int = 0) -> WeightDict:
    """Random engine weights (``[in, out]`` projections) with every key the DiT engine reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float16)

    def vec(n):
        return np.zeros(n, np.float16)

    weights = WeightDict()
    weights["scale_shift_table"] = mat(2, DIM)
    weights["proj_in.weight"], weights["proj_in.bias"] = mat(CHANNELS, DIM), vec(DIM)
    for name, (i, o) in {
        "time_embed.emb.timestep_embedder.linear_1": (256, DIM),
        "time_embed.emb.timestep_embedder.linear_2": (DIM, DIM),
        "time_embed.linear": (DIM, 6 * DIM),
        "caption_projection.linear_1": (TEXT_DIM, DIM),
        "caption_projection.linear_2": (DIM, DIM),
        "proj_out": (DIM, CHANNELS),
    }.items():
        weights[f"{name}.weight"], weights[f"{name}.bias"] = mat(i, o), vec(o)
    for layer in range(LAYERS):
        p = f"transformer_blocks.{layer}"
        weights[f"{p}.scale_shift_table"] = mat(6, DIM)
        for attn in ("attn1", "attn2"):
            weights[f"{p}.{attn}.norm_q.weight"] = np.ones(DIM, np.float32)
            weights[f"{p}.{attn}.norm_k.weight"] = np.ones(DIM, np.float32)
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                weights[f"{p}.{attn}.{proj}.weight"] = mat(DIM, DIM)
                weights[f"{p}.{attn}.{proj}.bias"] = vec(DIM)
        weights[f"{p}.ff.net.0.proj.weight"], weights[f"{p}.ff.net.0.proj.bias"] = (
            mat(DIM, FFN),
            vec(FFN),
        )
        weights[f"{p}.ff.net.2.weight"], weights[f"{p}.ff.net.2.bias"] = mat(FFN, DIM), vec(DIM)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(LAYERS)}


@pytest.fixture
def fake_scales(monkeypatch):
    """Use scales for the tiny shape instead of the committed table."""
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    return scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the weight within e4m3 precision."""
    weight = np.random.default_rng(1).normal(0.0, 0.3, (32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= float(np.abs(weight).max()) / 8


def test_all_zero_weight_is_rejected() -> None:
    """A weight with no positive amax has no scale to derive."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_quantized_projections_get_a_scale(fake_scales) -> None:
    """The feed-forward linears get a scale; attention, the embeddings and output layers do not."""
    ctx = calibrate_ltx(_weights(), quantization="fp8", **SHAPE)
    assert len(ctx.scales) == LAYERS * len(QUANTIZED) == 4
    assert ctx.should_quantize("transformer_blocks.1.ff.net.0.proj")
    assert ctx.should_quantize("transformer_blocks.0.ff.net.2")
    for other in (
        "transformer_blocks.1.attn1.to_k",
        "transformer_blocks.1.attn2.to_out.0",
        "proj_in",
        "proj_out",
        "time_embed.linear",
        "caption_projection.linear_1",
        "transformer_blocks.0.attn1.norm_q",
    ):
        assert not ctx.should_quantize(other)
    assert ctx.has_fp8_weights


def test_projection_without_scale_is_rejected(monkeypatch) -> None:
    """A projection with no activation scale fails loudly and names the calibration tool."""
    scales = _all_scales()
    del scales["transformer_blocks.0.ff.net.2"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    with pytest.raises(
        ValueError, match=r"transformer_blocks\.0\.ff\.net\.2.*calibrate_fp8_scales"
    ):
        calibrate_ltx(_weights(), quantization="fp8", **SHAPE)


def test_already_quantized_checkpoint_is_rejected(fake_scales) -> None:
    """A checkpoint that already stores quantization scales is not quantized a second time."""
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_ltx(_weights(), quantization="fp8", quantized_checkpoint=True, **SHAPE)
    readers = SimpleNamespace(tensor_map={"transformer_blocks.0.attn1.to_q.weight_scale": object()})
    assert is_quantized_checkpoint(readers)
    assert not is_quantized_checkpoint(SimpleNamespace(tensor_map={"proj_in.weight": 1}))


def test_missing_weight_is_rejected(fake_scales) -> None:
    """A scale for a projection the checkpoint lacks fails loudly."""
    weights = _weights()
    del weights["transformer_blocks.1.ff.net.2.weight"]
    with pytest.raises(ValueError, match="transformer_blocks.1.ff.net.2"):
        calibrate_ltx(weights, quantization="fp8", **SHAPE)


def test_unsupported_format_is_rejected() -> None:
    """Only fp8 is implemented."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_ltx(_weights(), quantization="nvfp4", **SHAPE)


def test_unknown_shape_has_a_clear_error() -> None:
    """A model shape with no committed scales names the shape and the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_ltx(_weights(), quantization="fp8", **{**SHAPE, "num_layers": 3})


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed model shape has a finite positive scale for each quantized projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        dim, layers, ffn = (int(part[1:]) for part in key.split("-")[1:])
        assert scale_key(dim, layers, ffn) == key
        assert set(scales) == set(quantized_names(layers)), key
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key


def test_configure_build_keeps_layer_names() -> None:
    """Quantized builds keep layer names so the kernels can be inspected."""
    config = SimpleNamespace(profiling_verbosity=None)
    configure_build(config)
    assert config.profiling_verbosity == quantization_module.trt.ProfilingVerbosity.DETAILED


def test_calibration_tool_maps_module_names_to_engine_names() -> None:
    """Diffusers module paths of the quantized projections map to themselves; other modules are skipped."""
    weight_name = calibrate_fp8_scales.weight_name
    for stem in QUANTIZED:
        assert weight_name(f"transformer_blocks.12.{stem}") == f"transformer_blocks.12.{stem}"
    for other in (
        "transformer_blocks.3.attn1.to_q",
        "transformer_blocks.3.attn2.to_out.0",
        "proj_in",
        "proj_out",
        "time_embed.linear",
        "caption_projection.linear_1",
        "transformer_blocks.3.ff.net.0",
        "transformer_blocks.3.attn1.to_q.base",
        "transformer_blocks.3.attn1.to_out",
    ):
        assert weight_name(other) is None


def test_calibration_tool_shape_config_matches_the_scale_key(tmp_path: Path) -> None:
    """The tool derives the same shape key as the build from the transformer config."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"num_attention_heads": 32, "attention_head_dim": 64, "num_layers": 28})
    )
    cfg = calibrate_fp8_scales.shape_config(tmp_path)
    key = scale_key(cfg.dim, cfg.num_layers, cfg.ffn_dim)
    assert key == scale_key(
        model._LTXVideoModel._DIT_DIM, model._LTXVideoModel._DIT_NUM_LAYERS, 4 * 2048
    )
    assert calibrate_fp8_scales.parse_size("480x704x17") == (480, 704, 17)


def test_calibration_prompts_are_many_and_distinct() -> None:
    """128 distinct prompts, none of them the premerge or the evaluation prompts."""
    prompts = calibrate_fp8_scales.calibration_prompts()
    assert len(prompts) == 128 == len(set(prompts))
    assert not any("red robot" in text or "lighthouse" in text for text in prompts)


def test_calibration_sizes_rotate_through_every_scene() -> None:
    """Every size is used equally often, and one scene meets every size across the styles."""
    sizes = ["64x64x9", "96x96x17", "128x128x25", "64x96x9"]
    assigned = calibrate_fp8_scales.assign_sizes(128, sizes)
    assert [assigned.count(size) for size in sizes] == [32] * 4
    scene = 5
    assert {assigned[scene + 32 * style] for style in range(4)} == set(sizes)


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(args=args))
    monkeypatch.setattr(
        "sys.argv",
        [
            "tool",
            "--model-dir",
            str(tmp_path),
            "--steps",
            "7",
            "--sizes",
            "64x64x9",
            "--prompts",
            "3",
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    calibrate_fp8_scales.main()
    assert seen["args"] == (tmp_path, 7, ["64x64x9"], 3, tmp_path / "report.json")


def _request(model_dir: Path, **overrides):
    values = {
        "model_dir": str(model_dir),
        "task": "image_generation",
        "precision": "fp16",
        "quantization": "fp8",
        "dynamic_kv_cache": False,
        "max_sequence_length": None,
        "video_num_frames": None,
        "context_parallel_size": 1,
        "tensor_parallel_size": 1,
        "fp32_layers": [],
        "mtp_seq_len": None,
        "image_height": None,
        "image_width": None,
        "max_batch_size": 1,
        "verbose": False,
        "backend": "trt",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Quantization formats other than fp8 are refused."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(_request(tmp_path, quantization="nvfp4"), Mock())


def test_build_rejects_fp8_with_fp32_precision(tmp_path: Path) -> None:
    """FP8 needs fp16."""
    with pytest.raises(NotImplementedError, match="fp16"):
        model.build(_request(tmp_path, precision="fp32"), Mock())


def _plan(quant_ctx, precision: str = "fp16") -> bytes:
    return dit.build_ltx_dit_engine(
        _weights(),
        latent_frames=1,
        latent_height=2,
        latent_width=2,
        text_seq_len=TEXT_LEN,
        text_dim=TEXT_DIM,
        in_channels=CHANNELS,
        dim=DIM,
        num_heads=HEADS,
        num_layers=LAYERS,
        precision=precision,
        quant_ctx=quant_ctx,
    )


def _build_dit(quant_ctx, precision: str = "fp16") -> int:
    return len(_plan(quant_ctx, precision))


def test_dit_builder_rejects_fp8_with_fp32_precision() -> None:
    """The denoiser builder refuses an FP8 context outside fp16."""
    with pytest.raises(NotImplementedError, match="fp16"):
        _build_dit(LTXQuantContext(scales={"transformer_blocks.0.ff.net.2": 0.02}), "fp32")


class _RecordingNetwork:
    """Records the layers a quantized projection adds, without building an engine."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append((name, args))
            return SimpleNamespace(get_output=lambda _index: SimpleNamespace(dtype="out"))

        return add


def test_quantized_projection_is_quantize_dequantize_matmul_in_the_activation_type() -> None:
    """Weight constant, two Dequantize in the activation type, one Quantize, one matmul, no cast."""
    ctx = LTXQuantContext(scales={"transformer_blocks.0.ff.net.2": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (32, 16)).astype(np.float32)
    ctx.matmul(network, SimpleNamespace(dtype="fp16"), weight, "transformer_blocks.0.ff.net.2")
    names = [name for name, _ in network.calls]
    assert names.count("add_dequantize") == 2
    assert names.count("add_quantize") == 1
    assert names.count("add_matrix_multiply") == 1
    assert "add_cast" not in names
    assert [args[-1] for name, args in network.calls if name == "add_dequantize"] == ["fp16"] * 2
    assert len(ctx.keep_alive) == 1 and ctx.keep_alive[0].shape == (32, 16)


class _SpyContext(LTXQuantContext):
    """Records which projections are quantized."""

    def __init__(self, scales):
        super().__init__(scales=scales)
        self.seen: dict[str, float] = {}

    def matmul(self, network, x, w_io, name):
        self.seen[name] = float(np.abs(w_io).max())
        return super().matmul(network, x, w_io, name)


def _run(plan: bytes, seed: int = 0) -> np.ndarray:
    """Run a tiny denoiser plan on random inputs and return its output."""
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "hidden_states": (1, 4, CHANNELS),
        "encoder_hidden_states": (1, TEXT_LEN, TEXT_DIM),
        "timestep": (1,),
        "encoder_attention_mask": (1, TEXT_LEN),
    }
    tensors = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            value = torch.randn(shapes[name], generator=generator)
            if name == "encoder_attention_mask":
                value = torch.ones(shapes[name])
            if name == "timestep":
                value = torch.full(shapes[name], 500.0)
        else:
            value = torch.empty(tuple(context.get_tensor_shape(name)))
        dtype = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }[engine.get_tensor_dtype(name)]
        tensors[name] = value.to(dtype).cuda()
        context.set_tensor_address(name, tensors[name].data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return tensors["sample"].float().cpu().numpy()


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_quantizes_only_the_feed_forward_and_is_smaller() -> None:
    """A tiny FP8 denoiser builds, quantizes only the feed-forward linears, and is smaller."""
    ctx = _SpyContext({name: 0.02 for name in quantized_names(LAYERS)})
    fp16_size = _build_dit(None)
    fp8_size = _build_dit(ctx)
    assert fp8_size < fp16_size
    assert set(ctx.seen) == set(quantized_names(LAYERS))
    key = "transformer_blocks.0.ff.net.0.proj"
    assert ctx.seen[key] == pytest.approx(float(np.abs(_weights()[f"{key}.weight"]).max()))


@pytest.mark.gpu
@pytest.mark.trt
def test_projection_without_scale_is_not_quantized_in_the_engine() -> None:
    """Only the projections that have a scale are quantized."""
    ctx = _SpyContext({"transformer_blocks.1.ff.net.0.proj": 0.02})
    _build_dit(ctx)
    assert set(ctx.seen) == {"transformer_blocks.1.ff.net.0.proj"}


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_output_is_finite_and_close_to_fp16() -> None:
    """The fp16 Dequantize path overflows to NaN when an output is too large: guard it.

    The tiny engine runs on random inputs; a large input scale and a large activation do not
    overflow. The real-size scan over all denoising steps of several prompts is in the PR.
    """
    scales = {name: 0.05 for name in quantized_names(LAYERS)}
    plain = _run(_plan(None))
    quantized = _run(_plan(LTXQuantContext(scales=scales)))
    assert np.isfinite(quantized).all()
    relative = np.linalg.norm(quantized - plain) / np.linalg.norm(plain)
    assert relative < 0.2
