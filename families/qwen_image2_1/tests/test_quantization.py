# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image 2.1 denoiser FP8 / NVFP4: recipe, packaged scales, weight packing, graphs, rejected configs."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ml_dtypes")

from families.qwen_image2_1 import calibrate_scales  # noqa: E402
from families.qwen_image2_1 import quantization as q  # noqa: E402

QWEN_IMAGE_21 = SimpleNamespace(num_layers=32, hidden=4096, mlp_ratio=3)
TINY = SimpleNamespace(num_layers=2, hidden=128, mlp_ratio=3)


def _weights(seed: int = 0, **extra) -> dict:
    rng = np.random.default_rng(seed)
    tensors = {
        "img_in.weight": rng.standard_normal((32, 16)).astype(np.float32),
        "proj_out.weight": rng.standard_normal((16, 32)).astype(np.float32),
    }
    tensors.update(extra)
    return tensors


def _scales(config, value=0.01) -> dict:
    return dict.fromkeys(q.projection_names(config.num_layers), value)


# ---------------------------------------------------------------- recipe


def test_fp8_recipe_is_every_block_projection_except_the_attention_output() -> None:
    names = q.projection_names(QWEN_IMAGE_21.num_layers)
    assert len(names) == len(set(names)) == 32 * 6
    assert not [name for name in names if "to_out" in name]
    assert {q.projection_kind(name) for name in names} == set(q.BLOCK_STEMS)
    with pytest.raises(ValueError, match="block projection"):
        q.projection_kind("txt_in.in_layer")


def test_nvfp4_recipe_is_the_swiglu_input_projections() -> None:
    assert set(q.NVFP4_KINDS) < set(q.BLOCK_STEMS)
    assert set(q.NVFP4_KINDS) == {"img_mlp.gate_layer", "img_mlp.proj"}


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_quantized_projections_follow_the_recipe(quantization: str) -> None:
    scales = _scales(QWEN_IMAGE_21)
    dropped = "transformer_blocks.3.img_mlp.out"
    scales[dropped] = None
    result = q.quantized_projections(quantization, QWEN_IMAGE_21, _weights(), scales)
    assert dropped not in result and len(result) == 191
    for name, (fmt, scale) in result.items():
        nvfp4 = quantization == "nvfp4" and q.projection_kind(name) in q.NVFP4_KINDS
        assert fmt == ("nvfp4" if nvfp4 else "fp8") and scale == 0.01
    count = sum(fmt == "nvfp4" for fmt, _ in result.values())
    assert count == (64 if quantization == "nvfp4" else 0)


def test_unknown_quantization_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantization"):
        q.quantized_projections("int8", QWEN_IMAGE_21, _weights(), _scales(QWEN_IMAGE_21))


@pytest.mark.parametrize(
    "extra",
    [
        {"transformer_blocks.0.attn.to_q.weight_scale": np.ones(1, np.float32)},
        {"transformer_blocks.0.attn.to_q.input_scale": np.ones(1, np.float32)},
        {"transformer_blocks.0.attn.to_q.scale_weight": np.ones(1, np.float32)},
        {"transformer_blocks.0.attn.to_q.weight": np.ones((2, 2), np.uint8)},
    ],
)
def test_an_already_quantized_checkpoint_is_refused(extra: dict) -> None:
    with pytest.raises(ValueError, match="already quantized"):
        q.quantized_projections("fp8", QWEN_IMAGE_21, _weights(**extra), _scales(QWEN_IMAGE_21))


def test_fp8_weights_are_refused_by_dtype() -> None:
    import ml_dtypes

    weights = _weights()
    weights["img_in.weight"] = weights["img_in.weight"].astype(ml_dtypes.float8_e4m3fn)
    with pytest.raises(ValueError, match="already quantized"):
        q.reject_quantized_checkpoint(weights)


# ---------------------------------------------------------------- scales


def test_packaged_scales_cover_the_qwen_image_21_recipe() -> None:
    payload = json.loads(q.SCALES_PATH.read_text(encoding="utf-8"))
    assert payload["model"] == q.model_key(QWEN_IMAGE_21) == "qwen_image2_1-l32-h4096-r3"
    assert len(payload["weight_abs_sums"]) == 2
    assert all(value > 0 for value in payload["weight_abs_sums"])
    scales = q.validated_scales(payload["scales"], QWEN_IMAGE_21)
    values = [value for value in scales.values() if value is not None]
    assert len(scales) == 192 and values
    # Calibrated inputs stay far from the BF16 range.
    assert all(1e-3 < value * q.FP8_E4M3_MAX < 1e5 for value in values)


def test_scales_of_another_checkpoint_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {
                "model": q.model_key(QWEN_IMAGE_21),
                "weight_abs_sums": q.weight_sums(weights),
                "scales": _scales(QWEN_IMAGE_21),
            }
        )
    )
    assert len(q.load_scales(QWEN_IMAGE_21, weights, path)) == 192
    with pytest.raises(ValueError, match="calibrate_scales"):
        q.load_scales(QWEN_IMAGE_21, _weights(seed=1), path)
    with pytest.raises(ValueError, match="calibrate_scales"):
        q.load_scales(TINY, weights, path)


def test_scales_must_cover_exactly_the_recipe() -> None:
    scales = _scales(TINY)
    assert len(q.validated_scales(scales, TINY)) == 12
    missing = dict(scales)
    missing.pop("transformer_blocks.1.img_mlp.proj")
    with pytest.raises(ValueError, match="missing"):
        q.validated_scales(missing, TINY)
    with pytest.raises(ValueError, match="unexpected"):
        q.validated_scales({**scales, "transformer_blocks.0.attn.to_out.0": 0.01}, TINY)
    with pytest.raises(TypeError, match="dictionary"):
        q.validated_scales([0.01], TINY)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan, True, "0.1"])
def test_scales_must_be_positive_finite_numbers(value) -> None:
    scales = _scales(TINY)
    scales["transformer_blocks.0.attn.to_q"] = value
    with pytest.raises(ValueError, match="positive and finite"):
        q.validated_scales(scales, TINY)


def test_calibration_keeps_outliers_bf16() -> None:
    amax = {f"transformer_blocks.{index}.img_mlp.out": 10.0 for index in range(5)}
    amax["transformer_blocks.5.img_mlp.out"] = 10.0 * q.OUTLIER_FACTOR + 1
    amax["transformer_blocks.0.attn.to_q"] = 0.0
    scales = calibrate_scales.scales_from_amax(amax)
    assert scales["transformer_blocks.5.img_mlp.out"] is None
    assert scales["transformer_blocks.0.attn.to_q"] is None
    assert scales["transformer_blocks.0.img_mlp.out"] == pytest.approx(10.0 / 448)


def test_calibration_runs_cover_both_manifest_configs() -> None:
    assert calibrate_scales.RUNS == ((1024, 40, 1.0), (512, 20, 4.0))


# ---------------------------------------------------------------- weights


def test_fp8_weight_scale_is_amax_over_448() -> None:
    weight = np.array([[0.5, -2.0], [1.0, 0.25]], dtype=np.float32)
    data, scale = q.fp8_weight(weight)
    assert scale == pytest.approx(2.0 / 448)
    assert np.allclose(data.astype(np.float32) * scale, weight, rtol=0.07)
    assert float(np.abs(data.astype(np.float32)).max()) == 448.0
    with pytest.raises(ValueError, match="positive finite"):
        q.fp8_weight(np.zeros((2, 2), np.float32))


def test_nvfp4_weight_layout_and_global_scale() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((8, 64)).astype(np.float32)
    packed, block_scale, weight_scale_2 = q.quantize_nvfp4_weight(weight)
    assert packed.shape == (8, 32) and packed.dtype == np.uint8
    assert block_scale.shape == (8, 4) and block_scale.dtype == np.uint8
    assert weight_scale_2 == pytest.approx(float(np.abs(weight).max()) / (6 * 448), rel=1e-6)
    restored = q.dequantize_nvfp4_weight(packed, block_scale, weight_scale_2)
    blocks = np.abs(weight).reshape(8, 4, 16).max(-1)
    # Error per value is at most half of the largest E2M1 gap (2) times the block scale.
    bound = np.repeat(blocks / 6 * 1.0 * 1.07, 16, axis=1)
    assert np.all(np.abs(restored - weight) <= bound + 1e-6)


def test_nvfp4_is_exact_on_the_grid_and_packs_the_first_value_low() -> None:
    pytest.importorskip("torch")
    row = np.array(
        [6.0, -0.5, 1.5, 0.0, 3.0, -6.0, 2.0, 4.0, 1.0, -1.0, 0.5, -3.0, -2.0, -4.0, -1.5, 0.0],
        dtype=np.float32,
    )
    packed, block_scale, weight_scale_2 = q.quantize_nvfp4_weight(row[None])
    # The block scale is 448 and 448 * weight_scale_2 is 1 up to one float32 rounding.
    assert np.allclose(
        q.dequantize_nvfp4_weight(packed, block_scale, weight_scale_2)[0], row, rtol=1e-6, atol=0
    )
    assert packed[0, 0] == (0x7 | (0x9 << 4))  # 6.0 low nibble, -0.5 high nibble


def test_nvfp4_all_zero_block_gets_block_scale_one() -> None:
    pytest.importorskip("torch")
    weight = np.zeros((1, 32), np.float32)
    weight[0, 16:] = np.linspace(-3, 3, 16)
    packed, block_scale, weight_scale_2 = q.quantize_nvfp4_weight(weight)
    restored = q.dequantize_nvfp4_weight(packed, block_scale, weight_scale_2)
    assert np.all(restored[0, :16] == 0) and np.isfinite(restored).all()


@pytest.mark.parametrize(
    "weight, message",
    [
        (np.ones((2, 24), np.float32), "multiple of 16"),
        (np.zeros((2, 16), np.float32), "positive finite"),
        (np.ones(16, np.float32), r"\[out, in\]"),
    ],
)
def test_nvfp4_rejects_unsupported_weights(weight: np.ndarray, message: str) -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match=message):
        q.quantize_nvfp4_weight(weight)


# ---------------------------------------------------------------- graphs

HEIGHT, WIDTH, TEXT_SLOTS = 4, 6, 40


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    root = tmp_path_factory.mktemp("tiny_qwen_image2_1")
    torch.manual_seed(0)
    model = diffusers.QwenImage21Transformer2DModel(
        in_channels=16,
        out_channels=16,
        num_layers=TINY.num_layers,
        attention_head_dim=32,
        num_attention_heads=4,
        context_in_dim=32,
        mlp_ratio=TINY.mlp_ratio,
        axes_dims_rope=(8, 12, 12),
    ).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            scale = 0.1 if parameter.ndim == 1 else 0.05
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    model.save_pretrained(root)
    return root


def _network(directory: Path, quantized: dict | None):
    """Layer types of the denoiser network, captured before TensorRT builds it."""
    trt = pytest.importorskip("tensorrt")
    from families.qwen_image2_1 import dit_builder, trt_ops

    config, weights = dit_builder.load_denoiser(directory)
    captured = {}

    def capture(graph, label):
        network = graph.network
        captured["kinds"] = [network.get_layer(i).type for i in range(network.num_layers)]
        captured["verbosity"] = graph.config.profiling_verbosity
        raise _Stop

    class _Stop(Exception):
        pass

    original = trt_ops.serialize
    trt_ops.serialize = capture
    try:
        with pytest.raises(_Stop):
            dit_builder.build_denoiser_plan(
                config,
                weights,
                io.BytesIO(),
                latent_height=HEIGHT,
                latent_width=WIDTH,
                text_tokens=TEXT_SLOTS,
                verbose=False,
                quantized=quantized,
            )
    finally:
        trt_ops.serialize = original
    return trt, captured


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_graph_quantizes_exactly_the_selected_projections(
    tiny_dir: Path, quantization: str
) -> None:
    pytest.importorskip("torch")
    spec = q.quantized_projections(quantization, TINY, _weights(), _scales(TINY))
    trt, captured = _network(tiny_dir, spec)
    kinds = captured["kinds"]
    fp8 = sum(fmt == "fp8" for fmt, _ in spec.values())
    nvfp4 = sum(fmt == "nvfp4" for fmt, _ in spec.values())
    # One activation quantization per block input it reads: attention input, SwiGLU input, SwiGLU product.
    blocks = TINY.num_layers
    if quantization == "fp8":
        assert kinds.count(trt.LayerType.QUANTIZE) == 3 * blocks
        assert kinds.count(trt.LayerType.DYNAMIC_QUANTIZE) == 0
        assert kinds.count(trt.LayerType.DEQUANTIZE) == 3 * blocks + fp8
    else:
        assert kinds.count(trt.LayerType.QUANTIZE) == 2 * blocks
        assert kinds.count(trt.LayerType.DYNAMIC_QUANTIZE) == blocks
        # FP8: one per activation and weight; NVFP4: two per activation and two per weight.
        assert kinds.count(trt.LayerType.DEQUANTIZE) == 2 * blocks + fp8 + 2 * blocks + 2 * nvfp4
    assert captured["verbosity"] == trt.ProfilingVerbosity.DETAILED


def test_unquantized_graph_is_unchanged(tiny_dir: Path) -> None:
    trt, plain = _network(tiny_dir, None)
    _, empty = _network(tiny_dir, {})
    assert trt.LayerType.QUANTIZE not in plain["kinds"]
    assert trt.LayerType.DYNAMIC_QUANTIZE not in plain["kinds"]
    assert plain["kinds"] == empty["kinds"]
    assert plain["verbosity"] != trt.ProfilingVerbosity.DETAILED


def test_quantized_build_needs_bf16(tiny_dir: Path) -> None:
    trt = pytest.importorskip("tensorrt")
    from families.qwen_image2_1.dit_builder import build_denoiser_plan, load_denoiser

    config, weights = load_denoiser(tiny_dir)
    with pytest.raises(ValueError, match="BF16"):
        build_denoiser_plan(
            config,
            weights,
            io.BytesIO(),
            latent_height=HEIGHT,
            latent_width=WIDTH,
            text_tokens=TEXT_SLOTS,
            verbose=False,
            compute=trt.float32,
            quantized=q.quantized_projections("fp8", TINY, _weights(), _scales(TINY)),
        )


def _engine_output(directory: Path, quantized: dict | None):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    from families.qwen_image2_1.dit_builder import build_denoiser_plan, load_denoiser

    config, weights = load_denoiser(directory)
    plan = io.BytesIO()
    build_denoiser_plan(
        config,
        weights,
        plan,
        latent_height=HEIGHT,
        latent_width=WIDTH,
        text_tokens=TEXT_SLOTS,
        verbose=False,
        quantized=quantized,
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan.getvalue())
    context = engine.create_execution_context()
    generator = torch.Generator().manual_seed(5)
    buffers = {
        "latents": torch.randn(1, HEIGHT * WIDTH, 16, generator=generator),
        "timestep": torch.tensor([0.6]),
        "text_hidden": torch.randn(1, TEXT_SLOTS, 32, generator=generator),
        "text_length": torch.tensor([17], dtype=torch.int32),
    }
    buffers = {name: value.cuda() for name, value in buffers.items()}
    buffers["velocity"] = torch.empty(1, HEIGHT * WIDTH, 16, device="cuda")
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return buffers["velocity"].cpu()


def _blackwell() -> bool:
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("quantization, bound", [("fp8", 0.05), ("nvfp4", 0.1)])
def test_quantized_engine_runs_close_to_bf16(
    tiny_dir: Path, quantization: str, bound: float
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if quantization == "nvfp4" and not _blackwell():
        pytest.skip("NVFP4 needs a Blackwell GPU")
    reference = _engine_output(tiny_dir, None)
    actual = _engine_output(
        tiny_dir, q.quantized_projections(quantization, TINY, _weights(), _scales(TINY, 0.02))
    )
    assert torch.isfinite(actual).all()
    error = float((actual - reference).norm() / reference.norm())
    print(f"tiny {quantization} relative L2 vs BF16: {error:.4f}")
    assert 0 < error < bound


# ---------------------------------------------------------------- build entry


class _Stop(Exception):
    pass


def _request(tmp_path: Path, quantization: str | None):
    from tensorrt_model_connect import BuildRequest

    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "x.bundle",
        family="qwen_image2_1",
        task="text_to_image",
        precision="bf16",
        quantization=quantization,
    )


@pytest.mark.parametrize("quantization, nvfp4", [("fp8", 0), ("nvfp4", 64), (None, None)])
def test_build_hands_the_recipe_to_the_denoiser_only(
    monkeypatch, tmp_path: Path, quantization, nvfp4
) -> None:
    from families.qwen_image2_1 import dit_builder, model, text_encoder_builder, vae_builder

    seen = {}
    monkeypatch.setattr(model, "scheduler_settings", lambda root: {"max_image_seq_len": 4096})
    monkeypatch.setattr(
        model,
        "prompt_layout",
        lambda root: {"frame_tokens": 8, "pad_token_id": 0, "prefix_tokens": 3},
    )
    (tmp_path / "processor").mkdir()
    (tmp_path / "processor" / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(model, "require_supported_tokenizer", lambda tokenizer: None)
    monkeypatch.setattr(model, "require_embedded_token_ids", lambda *args: None)
    monkeypatch.setattr(
        vae_builder, "load_vae", lambda root: (SimpleNamespace(spatial_scale=16, z_dim=64), {})
    )
    monkeypatch.setattr(
        text_encoder_builder,
        "load_text_encoder",
        lambda root: (SimpleNamespace(hidden=4096, vocab_size=10), {}),
    )
    monkeypatch.setattr(
        text_encoder_builder,
        "build_text_encoder_plan",
        lambda *args, **kwargs: seen.setdefault("text_encoder", kwargs),
    )
    monkeypatch.setattr(
        dit_builder,
        "load_denoiser",
        lambda root: (
            SimpleNamespace(
                num_layers=32,
                hidden=4096,
                mlp_ratio=3,
                context_dim=4096,
                in_channels=64,
                out_channels=64,
            ),
            _weights(),
        ),
    )
    monkeypatch.setattr(q, "load_scales", lambda config, weights: _scales(config))

    def denoiser(*args, **kwargs):
        seen["quantized"] = kwargs.get("quantized")
        raise _Stop

    monkeypatch.setattr(dit_builder, "build_denoiser_plan", denoiser)

    class _Section:
        def __enter__(self):
            return io.BytesIO()

        def __exit__(self, *exc):
            return False

    writer = SimpleNamespace(set_header=lambda **kw: None, open_section=lambda name: _Section())
    with pytest.raises(_Stop):
        model.build(_request(tmp_path, quantization), writer)
    assert "quantized" not in seen["text_encoder"]
    if quantization is None:
        assert seen["quantized"] is None
    else:
        assert len(seen["quantized"]) == 192
        assert sum(fmt == "nvfp4" for fmt, _ in seen["quantized"].values()) == nvfp4


def test_build_rejects_other_quantizations(tmp_path: Path) -> None:
    from families.qwen_image2_1 import model

    with pytest.raises(NotImplementedError, match="none, fp8 or nvfp4"):
        model.build(_request(tmp_path, "int8"), SimpleNamespace())
