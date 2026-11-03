# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""stable_diffusion FP8 self-quantization: weight scales, the scale table, the tool, the build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales, model
from .. import quantization as quantization_module
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    UNetQuantContext,
    configure_build,
    is_quantized,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
    unet_quant_context,
)

# A tiny UNet with the SDXL structure: no attention in the first level, linear projections,
# text_time conditioning, a shortcut convolution wherever a resnet changes its width.
CHANNELS, HEADS, GROUPS, CROSS, CONTEXT, POOLED, TIME_WIDTH, LATENT = (
    (32, 64),
    (2, 4),
    8,
    32,
    8,
    16,
    8,
    8,
)
TEMB = 4 * CHANNELS[0]
CFG = {
    "in_channels": 4,
    "sample_size": LATENT,
    "block_out_channels": list(CHANNELS),
    "layers_per_block": 1,
    "attention_head_dim": list(HEADS),
    "cross_attention_dim": CROSS,
    "norm_num_groups": GROUPS,
    "context_length": CONTEXT,
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": TIME_WIDTH,
    "addition_time_ids": 6,
    "pooled_projection_dim": POOLED,
    "projection_class_embeddings_input_dim": POOLED + 6 * TIME_WIDTH,
}


def _is_excluded(name: str) -> bool:
    """The public exclusion list, written out independently of the module's regexes."""
    return (
        name in ("conv_in", "conv_out")
        or name.startswith(("time_embedding.", "add_embedding."))
        or name.endswith((".conv_shortcut", ".time_emb_proj"))
        or (".resnets." in name and name.endswith(".conv2"))
    )


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random fp16 UNet weights with every key the builder reads (diffusers names)."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}

    def dense(name, out, inp, bias=True):
        weights[f"{name}.weight"] = rng.normal(0.0, 0.05, (out, inp)).astype(np.float16)
        if bias:
            weights[f"{name}.bias"] = rng.normal(0.0, 0.02, (out,)).astype(np.float16)

    def conv(name, out, inp, kernel=3):
        weights[f"{name}.weight"] = rng.normal(0.0, 0.05, (out, inp, kernel, kernel)).astype(
            np.float16
        )
        weights[f"{name}.bias"] = rng.normal(0.0, 0.02, (out,)).astype(np.float16)

    def norm(name, width):
        weights[f"{name}.weight"] = np.ones(width, np.float16)
        weights[f"{name}.bias"] = np.zeros(width, np.float16)

    def resnet(name, inp, out):
        norm(f"{name}.norm1", inp)
        conv(f"{name}.conv1", out, inp)
        dense(f"{name}.time_emb_proj", out, TEMB)
        norm(f"{name}.norm2", out)
        conv(f"{name}.conv2", out, out)
        if inp != out:
            conv(f"{name}.conv_shortcut", out, inp, kernel=1)

    def attention(name, width):
        norm(f"{name}.norm", width)
        dense(f"{name}.proj_in", width, width)
        block = f"{name}.transformer_blocks.0"
        for index in (1, 2, 3):
            norm(f"{block}.norm{index}", width)
        for attn, source in (("attn1", width), ("attn2", CROSS)):
            dense(f"{block}.{attn}.to_q", width, width, bias=False)
            dense(f"{block}.{attn}.to_k", width, source, bias=False)
            dense(f"{block}.{attn}.to_v", width, source, bias=False)
            dense(f"{block}.{attn}.to_out.0", width, width)
        dense(f"{block}.ff.net.0.proj", 8 * width, width)
        dense(f"{block}.ff.net.2", width, 4 * width)
        dense(f"{name}.proj_out", width, width)

    c0, c1 = CHANNELS
    dense("time_embedding.linear_1", TEMB, c0)
    dense("time_embedding.linear_2", TEMB, TEMB)
    dense("add_embedding.linear_1", TEMB, POOLED + 6 * TIME_WIDTH)
    dense("add_embedding.linear_2", TEMB, TEMB)
    conv("conv_in", c0, 4)
    resnet("down_blocks.0.resnets.0", c0, c0)
    conv("down_blocks.0.downsamplers.0.conv", c0, c0)
    resnet("down_blocks.1.resnets.0", c0, c1)
    attention("down_blocks.1.attentions.0", c1)
    resnet("mid_block.resnets.0", c1, c1)
    attention("mid_block.attentions.0", c1)
    resnet("mid_block.resnets.1", c1, c1)
    resnet("up_blocks.0.resnets.0", c1 + c1, c1)
    attention("up_blocks.0.attentions.0", c1)
    resnet("up_blocks.0.resnets.1", c1 + c0, c1)
    attention("up_blocks.0.attentions.1", c1)
    conv("up_blocks.0.upsamplers.0.conv", c1, c1)
    resnet("up_blocks.1.resnets.0", c1 + c0, c0)
    resnet("up_blocks.1.resnets.1", c0 + c0, c0)
    norm("conv_norm_out", c0)
    conv("conv_out", 4, c0)
    return weights


def _excluded_names(weights) -> list[str]:
    return [
        key[: -len(".weight")]
        for key in weights
        if key.endswith(".weight") and _is_excluded(key[: -len(".weight")])
    ]


def _all_scales(weights, value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(weights)}


def _sdxl_cfg(**overrides) -> dict:
    cfg = {
        "image_size": 512,
        "scheduler_class": "EulerAncestralDiscreteScheduler",
        "timestep_spacing": "trailing",
        "text_encoder_2": {"hidden_size": 1280},
        "unet": {
            "addition_embed_type": "text_time",
            "block_out_channels": [320, 640, 1280],
            "layers_per_block": 2,
            "cross_attention_dim": 2048,
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def tiny_scales(monkeypatch):
    """Use scales for the tiny UNet instead of the committed table."""
    scales = _all_scales(_tiny_weights())
    monkeypatch.setattr(quantization_module, "scale_key", lambda cfg: "tiny")
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    return scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the weight within e4m3 precision."""
    weight = np.random.default_rng(1).normal(0.0, 0.3, (16, 8, 3, 3)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= float(np.abs(weight).max()) / 8


def test_all_zero_weight_is_rejected() -> None:
    """A weight with no positive amax has no scale to derive."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_excluded_layers_stay_unquantized_and_every_other_conv_and_linear_is_covered() -> None:
    """The public exclusion list stays fp16; every other Conv2d and Linear gets FP8."""
    weights = _tiny_weights()
    names = set(quantized_names(weights))
    layers = {key[: -len(".weight")] for key, value in weights.items() if value.ndim in (2, 4)}
    excluded = {name for name in layers if _is_excluded(name)}
    # embeddings and conv_in/out, time_emb_proj, conv_shortcut, resnet conv2
    assert len(excluded) == 6 + 8 + 5 + 8
    assert names == layers - excluded
    # A few kinds that must be in: the resnet convolutions, resampling convolutions, attention
    # projections (linear and fused geglu) and the feed-forward output.
    for name in (
        "down_blocks.0.resnets.0.conv1",
        "up_blocks.1.resnets.1.conv1",
        "down_blocks.0.downsamplers.0.conv",
        "up_blocks.0.upsamplers.0.conv",
        "mid_block.attentions.0.proj_in",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2",
    ):
        assert name in names


@pytest.mark.parametrize(
    ("name", "rank", "expected"),
    [
        ("conv_in", 4, False),
        ("conv_out", 4, False),
        ("down_blocks.1.resnets.0.conv_shortcut", 4, False),
        ("down_blocks.1.resnets.0.time_emb_proj", 2, False),
        ("time_embedding.linear_1", 2, False),
        ("add_embedding.linear_2", 2, False),
        ("down_blocks.1.resnets.0.conv2", 4, False),
        ("mid_block.resnets.1.conv2", 4, False),
        ("down_blocks.0.downsamplers.0.conv", 4, True),
        ("down_blocks.1.resnets.0.conv1", 4, True),
        ("conv_in_extra.conv1", 4, True),
        ("mid_block.resnets.0.norm1", 1, False),
    ],
)
def test_layer_selection_is_anchored_on_module_names(name: str, rank: int, expected: bool) -> None:
    """Exclusions match whole module names, not substrings of unrelated names."""
    assert is_quantized(name, rank) is expected


def test_quant_context_covers_all_included_layers(tiny_scales) -> None:
    """The context keeps one activation scale per included layer and none for excluded ones."""
    ctx = unet_quant_context(_tiny_weights(), {}, quantization="fp8")
    assert set(ctx.scales) == set(tiny_scales)
    assert ctx.should_quantize("up_blocks.1.resnets.0.conv1")
    assert not any(ctx.should_quantize(name) for name in _excluded_names(_tiny_weights()))


def test_included_layer_without_scale_is_rejected(monkeypatch, tiny_scales) -> None:
    """A layer the recipe quantizes but the scale file lacks fails loudly, not silently fp16."""
    scales = dict(tiny_scales)
    del scales["down_blocks.0.resnets.0.conv1"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match="no activation scale"):
        unet_quant_context(_tiny_weights(), {}, quantization="fp8")


def test_scale_for_unknown_layer_is_rejected(monkeypatch, tiny_scales) -> None:
    """A scale for a layer that is excluded or absent means the table and recipe disagree."""
    monkeypatch.setattr(
        quantization_module, "load_activation_scales", lambda key: {**tiny_scales, "conv_in": 0.1}
    )
    with pytest.raises(ValueError, match="does not quantize"):
        unet_quant_context(_tiny_weights(), {}, quantization="fp8")


def test_non_positive_scale_is_rejected(monkeypatch, tiny_scales) -> None:
    """Scales must be finite and positive."""
    scales = dict(tiny_scales)
    scales["mid_block.resnets.0.conv1"] = 0.0
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda key: scales)
    with pytest.raises(ValueError, match="positive"):
        unet_quant_context(_tiny_weights(), {}, quantization="fp8")


def test_unsupported_format_is_rejected(tiny_scales) -> None:
    """Only fp8 is implemented."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        unet_quant_context(_tiny_weights(), {}, quantization="nvfp4")


def test_scale_key_names_model_and_size() -> None:
    """The SDXL-Turbo UNet is keyed by image size."""
    assert scale_key(_sdxl_cfg()) == "sdxl-turbo-s512"
    assert scale_key(_sdxl_cfg(image_size=1024)) == "sdxl-turbo-s1024"


def test_sd15_and_other_unets_are_refused() -> None:
    """SD 1.5 (no second text encoder) and other UNet configs have no calibrated scales."""
    sd15 = _sdxl_cfg(text_encoder_2=None)
    sd15["unet"] = {
        **sd15["unet"],
        "addition_embed_type": None,
        "block_out_channels": [320, 640, 1280, 1280],
    }
    with pytest.raises(NotImplementedError, match="SDXL-Turbo UNet only"):
        scale_key(sd15)
    odd = _sdxl_cfg()
    odd["unet"] = {**odd["unet"], "block_out_channels": [128, 256, 512]}
    with pytest.raises(NotImplementedError, match="SDXL-Turbo UNet only"):
        scale_key(odd)


@pytest.mark.parametrize(
    "scheduler",
    [
        {"scheduler_class": "EulerDiscreteScheduler", "timestep_spacing": "leading"},
        {"scheduler_class": "EulerDiscreteScheduler", "timestep_spacing": "trailing"},
        {"scheduler_class": "", "timestep_spacing": ""},
    ],
)
def test_other_sdxl_variants_are_refused(scheduler: dict) -> None:
    """SDXL base and Lightning share the UNet config but are not the calibrated checkpoint."""
    with pytest.raises(NotImplementedError, match="SDXL base and Lightning"):
        scale_key(_sdxl_cfg(**scheduler))


def test_unknown_image_size_has_a_clear_error() -> None:
    """A size with no committed scales names the key and the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        quantization_module.load_activation_scales(scale_key(_sdxl_cfg(image_size=768)))


def test_committed_scales_cover_every_supported_size() -> None:
    """Each committed size has a finite positive scale for every quantized layer, and no other."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert sorted(table) == ["sdxl-turbo-s1024", "sdxl-turbo-s512"]
    for key, scales in table.items():
        size = int(key.rsplit("-s", 1)[1])
        assert scale_key(_sdxl_cfg(image_size=size)) == key
        # SDXL-Turbo UNet: every Conv2d and Linear except the excluded ones carries a scale.
        assert len(scales) == EXPECTED_SDXL_TURBO_LAYERS, key
        assert all(is_quantized(name, 2) for name in scales), key
        assert all(np.isfinite(value) and value > 0 for value in scales.values()), key
        kinds = {name.rsplit(".", 1)[-1] for name in scales}
        assert {"conv1", "to_q", "proj_in", "proj"} <= kinds and "conv2" not in kinds, key


# Counted from the SDXL-Turbo UNet checkpoint by the calibration tool.
EXPECTED_SDXL_TURBO_LAYERS = 743


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
    config = _FakeBuilderConfig(routes)
    configure_build(config)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None


def test_calibration_tool_picks_a_stable_prompt_set(tmp_path: Path) -> None:
    """Prompts are filtered, de-duplicated and shuffled with a fixed seed."""
    tsv = tmp_path / "prompts.tsv"
    rows = ["Prompt\tCategory"] + [f"a photo of subject number {i}\tx" for i in range(20)]
    rows += ["short\tx", "a photo of subject number 3\tx"]
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    first = calibrate_fp8_scales.select_prompts(tsv, 8)
    assert first == calibrate_fp8_scales.select_prompts(tsv, 8)
    assert len(first) == len(set(first)) == 8 and "short" not in first
    assert calibrate_fp8_scales.select_prompts(tsv, 4) == first[:4]
    with pytest.raises(ValueError, match="--prompts"):
        calibrate_fp8_scales.select_prompts(tsv, 0)


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(args=args))
    monkeypatch.setattr(
        "sys.argv",
        ["tool", "--model-dir", str(tmp_path), "--sizes", "512", "--steps", "2", "--prompts", "3"],
    )
    calibrate_fp8_scales.main()
    assert seen["args"] == (tmp_path, [512], 2, 3)


def _write_components(root: Path, *, xl: bool, scheduler: dict) -> Path:
    """Component configs of the real SDXL / SD 1.5 UNets; no weights."""
    unet = {
        "in_channels": 4,
        "block_out_channels": [320, 640, 1280] if xl else [320, 640, 1280, 1280],
        "layers_per_block": 2,
        "attention_head_dim": [5, 10, 20] if xl else 8,
        "cross_attention_dim": 2048 if xl else 768,
        "addition_embed_type": "text_time" if xl else None,
        "addition_time_embed_dim": 256 if xl else None,
        "projection_class_embeddings_input_dim": 2816 if xl else None,
        "use_linear_projection": xl,
    }
    text = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 77,
    }
    components = {
        "unet": unet,
        "vae": {
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "latent_channels": 4,
        },
        "text_encoder": text,
        "scheduler": scheduler,
    }
    if xl:
        components["text_encoder_2"] = {**text, "hidden_size": 1280, "projection_dim": 1280}
    (root / "model_index.json").write_text("{}")
    for name, payload in components.items():
        (root / name).mkdir()
        file = "scheduler_config.json" if name == "scheduler" else "config.json"
        (root / name / file).write_text(json.dumps(payload))
    return root


TURBO_SCHEDULER = {"_class_name": "EulerAncestralDiscreteScheduler", "timestep_spacing": "trailing"}


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
        "image_height": 512,
        "image_width": 512,
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


def test_build_rejects_fp8_outside_fp16(tmp_path: Path) -> None:
    """FP8 needs fp16 precision."""
    with pytest.raises(NotImplementedError, match="fp16"):
        model.build(_request(tmp_path, precision="fp32"), Mock())


def test_build_refuses_sd15_before_reading_weights(tmp_path: Path) -> None:
    """SD 1.5 has no calibrated scales: the error names the supported model, and no weight is read."""
    root = _write_components(tmp_path, xl=False, scheduler={"_class_name": "PNDMScheduler"})
    with pytest.raises(NotImplementedError, match="SDXL-Turbo UNet only"):
        model.build(_request(root), Mock())


def test_build_refuses_sdxl_base_before_reading_weights(tmp_path: Path) -> None:
    """SDXL base has the Turbo UNet config, with other weights and another scheduler."""
    base = {"_class_name": "EulerDiscreteScheduler", "timestep_spacing": "leading"}
    root = _write_components(tmp_path, xl=True, scheduler=base)
    with pytest.raises(NotImplementedError, match="SDXL base and Lightning"):
        model.build(_request(root), Mock())


def test_build_refuses_an_uncalibrated_image_size(tmp_path: Path) -> None:
    """SDXL-Turbo at a size with no committed scales points at the calibration tool."""
    root = _write_components(tmp_path, xl=True, scheduler=TURBO_SCHEDULER)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        model.build(_request(root, image_height=768, image_width=768), Mock())


def test_unet_builder_rejects_fp8_with_fp32_precision() -> None:
    """The UNet engine builder refuses an FP8 context outside fp16."""
    with pytest.raises(NotImplementedError, match="fp16"):
        model.build_unet_engine(
            _tiny_weights(),
            CFG,
            precision="fp32",
            quant=UNetQuantContext({"mid_block.resnets.0.conv1": 0.02}),
        )


class _RecordingNetwork:
    """Records the layers a quantized layer adds, without building an engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append(name)
            output = SimpleNamespace(dtype="fp16", shape=(1, 4, 4, 4))
            return SimpleNamespace(get_output=lambda _index: output, set_input=lambda *a: None)

        return add


def test_quantized_linear_is_quantize_dequantize_matmul_cast_bias() -> None:
    """A linear adds the FP8 weight constant, two Dequantize, one Quantize, a matmul, a cast and the bias."""
    ctx = UNetQuantContext({"attn.to_q": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (16, 32)).astype(np.float16)
    x = SimpleNamespace(dtype="half", shape=(1, 8, 32))
    ctx.linear(network, x, weight, np.zeros(16, np.float16), "attn.to_q")
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_matrix_multiply") == 1
    assert "add_cast" in network.calls and network.calls[-1] == "add_elementwise"
    assert ctx.keep_alive[0].shape == (1, 32, 16) and ctx.keep_alive[0].dtype == np.uint8


def test_quantized_conv_takes_the_fp8_weight_as_its_kernel_input() -> None:
    """A conv adds Quantize and two Dequantize around a convolution that reads its kernel from a layer."""
    ctx = UNetQuantContext({"block.conv1": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(3).normal(0.0, 0.1, (16, 8, 3, 3)).astype(np.float16)
    x = SimpleNamespace(dtype="half", shape=(1, 8, 4, 4))
    ctx.conv2d(network, x, weight, np.zeros(16, np.float16), "block.conv1", padding=(1, 1))
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_convolution_nd") == 1
    assert network.calls[-1] == "add_cast"
    assert ctx.keep_alive[0].shape == (16, 8, 3, 3)


class _SpyContext(UNetQuantContext):
    """Records which layers the builder routes through FP8."""

    def __init__(self, scales):
        super().__init__(scales)
        self.seen: dict[str, str] = {}

    def linear(self, network, x, weight, bias, name, **kwargs):
        self.seen[name] = "linear"
        return super().linear(network, x, weight, bias, name, **kwargs)

    def conv2d(self, network, x, weight, bias, name, **kwargs):
        self.seen[name] = "conv"
        return super().conv2d(network, x, weight, bias, name, **kwargs)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_with_the_expected_fp8_layers() -> None:
    """The tiny FP8 UNet builds and routes every included layer through FP8.

    Plan size is not asserted: at this size the kernels and tactic data outweigh the weights, so
    the FP8 plan can be larger than the fp16 one (it is on some GPUs). The size gain is measured
    on the real model.
    """
    import tensorrt as trt

    weights = _tiny_weights()
    ctx = _SpyContext(_all_scales(weights))
    fp8_plan = model.build_unet_engine(weights, CFG, precision="fp16", quant=ctx)
    assert set(ctx.seen) == set(ctx.scales)
    assert ctx.seen["down_blocks.0.resnets.0.conv1"] == "conv"
    assert ctx.seen["mid_block.attentions.0.transformer_blocks.0.attn1.to_q"] == "linear"
    assert not set(ctx.seen) & set(_excluded_names(weights))

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8_plan)
    info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    assert "FP8" in json.dumps(info), "the plan has no FP8 tensors"
    assert [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)].count(
        "out_sample"
    ) == 1


def _network_layers(quant) -> list[tuple[str, str]]:
    """Names and types of every layer the UNet builder adds to a TensorRT network."""
    import tensorrt as trt

    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    model.unet_builder.build_unet(
        network, _tiny_weights(), CFG, np.float16, trt.float16, quant=quant
    )
    return [
        (network.get_layer(i).name, str(network.get_layer(i).type))
        for i in range(network.num_layers)
    ]


@pytest.mark.gpu
@pytest.mark.trt
def test_quantization_selecting_nothing_leaves_the_network_unchanged() -> None:
    """A context that selects no layer builds the same network as no context at all."""
    assert _network_layers(UNetQuantContext({})) == _network_layers(None)
