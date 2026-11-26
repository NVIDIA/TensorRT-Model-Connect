# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chroma DiT FP8 / NVFP4: recipe, packaged scales, weight packing, graph selection, rejected configs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ml_dtypes")

from families.chroma import calibrate_scales  # noqa: E402
from families.chroma import quantization as q  # noqa: E402
from families.chroma.checkpoint import TransformerConfig  # noqa: E402

CHROMA1_HD = TransformerConfig(
    num_layers=19,
    num_single_layers=38,
    num_heads=24,
    head_dim=128,
    in_channels=64,
    joint_attention_dim=4096,
    axes_dims_rope=(16, 56, 56),
    approximator_channels=64,
    approximator_hidden=5120,
    approximator_layers=5,
)
TINY = TransformerConfig(
    num_layers=2,
    num_single_layers=2,
    num_heads=4,
    head_dim=64,
    in_channels=64,
    joint_attention_dim=256,
    axes_dims_rope=(16, 24, 24),
    approximator_channels=64,
    approximator_hidden=128,
    approximator_layers=2,
)


class _Weights:
    """The SafetensorsDirectory surface the recipe reads."""

    def __init__(
        self, tensors: dict[str, np.ndarray], dtypes: dict[str, str] | None = None
    ) -> None:
        self.tensors = tensors
        self.dtypes = dtypes or {}

    def names(self) -> set[str]:
        return set(self.tensors)

    def dtype(self, name: str) -> str:
        return self.dtypes.get(name, "BF16")

    def __getitem__(self, name: str) -> np.ndarray:
        return self.tensors[name]


def _weights(seed: int = 0, **extra) -> _Weights:
    rng = np.random.default_rng(seed)
    tensors = {
        "x_embedder.weight": rng.standard_normal((32, 64)).astype(np.float32),
        "proj_out.weight": rng.standard_normal((64, 32)).astype(np.float32),
    }
    tensors.update(extra)
    return _Weights(tensors)


def _scales(config, value=0.01) -> dict:
    return dict.fromkeys(q.projection_names(config.num_layers, config.num_single_layers), value)


# ---------------------------------------------------------------- recipe


def test_fp8_recipe_is_every_block_projection_except_the_attention_outputs() -> None:
    names = q.projection_names(CHROMA1_HD.num_layers, CHROMA1_HD.num_single_layers)
    assert len(names) == len(set(names)) == 19 * 10 + 38 * 5 == 380
    assert not [name for name in names if "to_out" in name or "to_add_out" in name]
    assert not [
        name for name in names if name.endswith("proj_out")
    ]  # only the MLP half of proj_out
    assert sum(name.endswith("proj_out.mlp") for name in names) == 38
    assert {q.projection_kind(name) for name in names} == {
        *(f"transformer_blocks.{stem}" for stem in q.DOUBLE_IMAGE_STEMS + q.DOUBLE_TEXT_STEMS),
        *(f"single_transformer_blocks.{stem}" for stem in q.SINGLE_STEMS),
    }


def test_nvfp4_recipe_is_a_subset_of_the_fp8_recipe() -> None:
    kinds = {q.projection_kind(name) for name in q.projection_names(1, 1)}
    assert set(q.NVFP4_KINDS) <= kinds


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_quantized_projections_follow_the_recipe(quantization: str) -> None:
    scales = _scales(CHROMA1_HD)
    dropped = "transformer_blocks.3.ff.net.2"
    scales[dropped] = None
    result = q.quantized_projections(quantization, CHROMA1_HD, _weights(), scales)
    assert dropped not in result and len(result) == 379
    for name, (fmt, scale) in result.items():
        expected = (
            "nvfp4"
            if quantization == "nvfp4" and q.projection_kind(name) in q.NVFP4_KINDS
            else "fp8"
        )
        assert fmt == expected and scale == 0.01
    count = sum(fmt == "nvfp4" for fmt, _ in result.values())
    if quantization == "nvfp4":
        assert count == sum(q.projection_kind(name) in q.NVFP4_KINDS for name in result) > 0
    else:
        assert count == 0


def test_unknown_quantization_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantization"):
        q.quantized_projections("int8", CHROMA1_HD, _weights(), _scales(CHROMA1_HD))


@pytest.mark.parametrize(
    "extra",
    [
        {"transformer_blocks.0.attn.to_q.weight_scale": np.ones(1, np.float32)},
        {"transformer_blocks.0.attn.to_q.input_scale": np.ones(1, np.float32)},
        {"transformer_blocks.0.attn.to_q.scale_weight": np.ones(1, np.float32)},
    ],
)
def test_an_already_quantized_checkpoint_is_refused(extra: dict) -> None:
    with pytest.raises(ValueError, match="already quantized"):
        q.quantized_projections("fp8", CHROMA1_HD, _weights(**extra), _scales(CHROMA1_HD))


def test_fp8_weights_are_refused_by_dtype() -> None:
    weights = _weights()
    weights.dtypes["x_embedder.weight"] = "F8_E4M3"
    with pytest.raises(ValueError, match="already quantized"):
        q.reject_quantized_checkpoint(weights)


# ---------------------------------------------------------------- scales


def test_packaged_scales_cover_the_chroma1_hd_recipe() -> None:
    payload = json.loads(q.SCALES_PATH.read_text(encoding="utf-8"))
    assert payload["model"] == q.model_key(CHROMA1_HD) == "chroma-l19-s38-d3072"
    assert len(payload["weight_abs_sums"]) == 2 and all(
        value > 0 for value in payload["weight_abs_sums"]
    )
    scales = q.validated_scales(payload["scales"], CHROMA1_HD)
    values = [value for value in scales.values() if value is not None]
    assert len(scales) == 380 and values
    # Calibrated inputs stay far from the BF16 range and within the outlier rule.
    assert all(1e-3 < value * q.FP8_E4M3_MAX < 1e3 for value in values)


def test_scales_of_another_checkpoint_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    path = tmp_path / "scales.json"
    path.write_text(
        json.dumps(
            {
                "model": q.model_key(CHROMA1_HD),
                "weight_abs_sums": q.weight_sums(weights),
                "scales": _scales(CHROMA1_HD),
            }
        )
    )
    assert len(q.load_scales(CHROMA1_HD, weights, path)) == 380
    with pytest.raises(ValueError, match="calibrate_scales"):
        q.load_scales(CHROMA1_HD, _weights(seed=1), path)
    with pytest.raises(ValueError, match="calibrate_scales"):
        q.load_scales(TINY, weights, path)


def test_scales_must_cover_exactly_the_recipe() -> None:
    scales = _scales(TINY)
    assert len(q.validated_scales(scales, TINY)) == 30
    missing = dict(scales)
    missing.pop("single_transformer_blocks.1.proj_mlp")
    with pytest.raises(ValueError, match="missing"):
        q.validated_scales(missing, TINY)
    with pytest.raises(ValueError, match="unexpected"):
        q.validated_scales({**scales, "transformer_blocks.0.attn.to_out.0": 0.01}, TINY)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan, True, "0.1"])
def test_scales_must_be_positive_finite_numbers(value) -> None:
    scales = _scales(TINY)
    scales["transformer_blocks.0.attn.to_q"] = value
    with pytest.raises(ValueError, match="positive and finite"):
        q.validated_scales(scales, TINY)


def test_calibration_keeps_outliers_bf16_and_reads_the_mlp_half_of_proj_out() -> None:
    amax = {f"transformer_blocks.{index}.ff.net.2": 10.0 for index in range(5)}
    amax["transformer_blocks.5.ff.net.2"] = 10.0 * q.OUTLIER_FACTOR + 1
    scales = calibrate_scales.scales_from_amax(amax)
    assert scales["transformer_blocks.5.ff.net.2"] is None
    assert scales["transformer_blocks.0.ff.net.2"] == pytest.approx(10.0 / 448)
    assert (
        calibrate_scales.hooked_name("single_transformer_blocks.3.proj_out.mlp")
        == "single_transformer_blocks.3.proj_out"
    )
    assert (
        calibrate_scales.hooked_name("transformer_blocks.3.attn.to_q")
        == "transformer_blocks.3.attn.to_q"
    )


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


@pytest.mark.parametrize(
    "weight, message",
    [
        (np.ones((2, 24), np.float32), "multiple of 16"),
        (np.zeros((2, 16), np.float32), "positive finite"),
        (np.ones(16, np.float32), r"\[out, in\]"),
    ],
)
def test_nvfp4_rejects_unsupported_weights(weight: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        q.quantize_nvfp4_weight(weight)


# ---------------------------------------------------------------- graphs


def _tiny_transformer(root: Path) -> Path:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    generator = torch.Generator().manual_seed(1)
    model = diffusers.ChromaTransformer2DModel(
        patch_size=1,
        in_channels=TINY.in_channels,
        num_layers=TINY.num_layers,
        num_single_layers=TINY.num_single_layers,
        attention_head_dim=TINY.head_dim,
        num_attention_heads=TINY.num_heads,
        joint_attention_dim=TINY.joint_attention_dim,
        axes_dims_rope=TINY.axes_dims_rope,
        approximator_num_channels=TINY.approximator_channels,
        approximator_hidden_dim=TINY.approximator_hidden,
        approximator_layers=TINY.approximator_layers,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    model.save_pretrained(root / "transformer")
    return root / "transformer"


def _network(directory: Path, quantized: dict | None):
    trt = pytest.importorskip("tensorrt")
    from families.chroma.checkpoint import SafetensorsDirectory
    from families.chroma.graph_ops import Graph
    from families.chroma.transformer_builder import build_transformer_network

    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, "bf16")
    build_transformer_network(graph, SafetensorsDirectory(directory), TINY, 32, 4, 6, quantized)
    kinds = [network.get_layer(index).type for index in range(network.num_layers)]
    return trt, network, graph, kinds


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    return _tiny_transformer(tmp_path_factory.mktemp("tiny_chroma"))


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_graph_quantizes_exactly_the_selected_projections(
    tiny_dir: Path, quantization: str
) -> None:
    spec = q.quantized_projections(quantization, TINY, _weights(), _scales(TINY))
    trt, _, _, kinds = _network(tiny_dir, spec)
    fp8 = sum(fmt == "fp8" for fmt, _ in spec.values())
    nvfp4 = sum(fmt == "nvfp4" for fmt, _ in spec.values())
    assert kinds.count(trt.LayerType.QUANTIZE) == fp8
    assert kinds.count(trt.LayerType.DYNAMIC_QUANTIZE) == nvfp4
    # FP8: activation and weight Dequantize; NVFP4: two for the weight, two for the activation.
    assert kinds.count(trt.LayerType.DEQUANTIZE) == 2 * fp8 + 4 * nvfp4


def test_unquantized_graph_is_unchanged(tiny_dir: Path) -> None:
    trt, network, _, kinds = _network(tiny_dir, None)
    assert trt.LayerType.QUANTIZE not in kinds and trt.LayerType.DYNAMIC_QUANTIZE not in kinds
    _, empty, _, empty_kinds = _network(tiny_dir, {})
    assert kinds == empty_kinds and network.num_layers == empty.num_layers


def _engine_output(directory: Path, quantized: dict | None):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    from families.chroma.checkpoint import SafetensorsDirectory
    from families.chroma.graph_ops import timestep_embedding
    from families.chroma.model import _plan
    from families.chroma.transformer_builder import build_transformer_network

    weights = SafetensorsDirectory(directory)
    plan = _plan(
        "transformer",
        "bf16",
        lambda graph: build_transformer_network(graph, weights, TINY, 32, 4, 6, quantized),
        False,
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    generator = torch.Generator().manual_seed(5)
    inputs = {
        "hidden_states": torch.randn(1, 24, 64, generator=generator),
        "encoder_hidden_states": torch.randn(1, 32, 256, generator=generator),
        "timestep_features": torch.from_numpy(
            timestep_embedding(np.array([500.0], np.float32), 16)
        ),
        "text_mask": torch.cat([torch.ones(1, 9), torch.zeros(1, 23)], dim=1),
    }
    buffers = {name: value.cuda() for name, value in inputs.items()}
    buffers["noise_pred"] = torch.empty(1, 24, 64, device="cuda")
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return buffers["noise_pred"].cpu()


def _blackwell() -> bool:
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("quantization, bound", [("fp8", 0.1), ("nvfp4", 0.3)])
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
        tiny_dir, q.quantized_projections(quantization, TINY, _weights(), _scales(TINY, 0.05))
    )
    assert torch.isfinite(actual).all()
    error = float((actual - reference).norm() / reference.norm())
    print(f"tiny {quantization} relative L2 vs BF16: {error:.4f}")
    assert 0 < error < bound


# ---------------------------------------------------------------- build entry


def test_build_hands_the_recipe_to_the_transformer_only(monkeypatch, tmp_path: Path) -> None:
    from families.chroma import model
    from tensorrt_model_connect import BuildRequest

    seen = {}

    class _Stop(Exception):
        pass

    def capture(writer, checkpoint, precision, verbose, height, width, length, quantized=None):
        seen["quantized"] = quantized
        raise _Stop

    checkpoint = SimpleNamespace(root=tmp_path, transformer=CHROMA1_HD)
    monkeypatch.setattr(model, "load_checkpoint", lambda root: checkpoint)
    monkeypatch.setattr(model, "SafetensorsDirectory", lambda directory: _weights())
    monkeypatch.setattr(q, "load_scales", lambda config, weights: _scales(config))
    monkeypatch.setattr("families.chroma.tokenizer_asset.tokenizer_json", lambda root: b"{}")
    monkeypatch.setattr(model, "_runtime_config", lambda *args: {})
    monkeypatch.setattr(model, "_write_components", capture)
    writer = SimpleNamespace(
        set_header=lambda **kw: None, add_json=lambda *a: None, add_bytes=lambda *a: None
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "x.bundle",
        family="chroma",
        task="text_to_image",
        precision="bf16",
        image_height=512,
        image_width=512,
        quantization="nvfp4",
    )
    with pytest.raises(_Stop):
        model.build(request, writer)
    assert len(seen["quantized"]) == 380
    assert {fmt for fmt, _ in seen["quantized"].values()} == {"fp8", "nvfp4"}
