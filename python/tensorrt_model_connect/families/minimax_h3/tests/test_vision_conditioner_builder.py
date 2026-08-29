# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3 import vision_conditioner_builder as vision


def _checkpoint_config() -> dict:
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": {"dtype": "bfloat16", "hidden_size": 5120},
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_vl",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 5120,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def _toy_spec() -> vision.MiniMaxH3VisionConditionerSpec:
    return vision.MiniMaxH3VisionConditionerSpec(
        image_height=4,
        image_width=8,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        depth=3,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=6,
        deepstack_visual_indexes=(0, 2),
    )


def _toy_ref2va_spec() -> vision.MiniMaxH3VisionConditionerSpec:
    return vision.MiniMaxH3VisionConditionerSpec.for_workflow(
        "ref2va",
        image_height=4,
        image_width=8,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        depth=3,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=6,
        deepstack_visual_indexes=(0, 2),
        min_patches=4,
        opt_patches=8,
        max_patches=16,
    )


def test_h3_processor_and_output_shapes_are_exact() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.from_checkpoint_config(_checkpoint_config())
    assert (spec.grid_t, spec.grid_h, spec.grid_w) == (1, 48, 84)
    assert spec.patch_vector_size == 3 * 2 * 16 * 16 == 1536
    assert spec.pixel_values_shape == (4032, 1536)
    assert spec.num_merged_tokens == 1008
    assert spec.output_shape == (1008, 5120)
    assert spec.deepstack_visual_indexes == (8, 16, 24)


def test_ref2va_vision_profile_is_dynamic_and_reuses_the_same_weights() -> None:
    config = _checkpoint_config()
    fl2va = vision.MiniMaxH3VisionConditionerSpec.from_checkpoint_config(config)
    ref2va = vision.MiniMaxH3VisionConditionerSpec.from_checkpoint_config(config, workflow="ref2va")
    assert fl2va.workflow == "fl2va"
    assert (fl2va.min_patches, fl2va.opt_patches, fl2va.max_patches) == (
        4032,
        4032,
        4032,
    )
    assert ref2va.workflow == "ref2va"
    assert (ref2va.min_patches, ref2va.opt_patches, ref2va.max_patches) == (
        2304,
        4032,
        65536,
    )
    assert vision.expected_weight_shapes(fl2va) == vision.expected_weight_shapes(ref2va)
    image = vision._specialize_ref2va_spec(ref2va, "image")
    video = vision._specialize_ref2va_spec(ref2va, "video")
    assert (image.min_patches, image.opt_patches, image.max_patches) == (
        16384,
        16384,
        65536,
    )
    assert (video.min_patches, video.opt_patches, video.max_patches) == (2304, 4032, 4176)


def test_ref2va_modality_precision_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="must be 'image' or 'video'"):
        vision.build_vision_conditioner_engine(
            _checkpoint_config(),
            {},
            workflow="ref2va",
            ref2va_modality="audio",
        )
    with pytest.raises(ValueError, match="must be 'image' or 'video'"):
        vision.build_vision_conditioner_engine(
            _checkpoint_config(),
            {},
            workflow="ref2va",
        )
    with pytest.raises(ValueError, match="FL2VA does not accept"):
        vision.build_vision_conditioner_engine(
            _checkpoint_config(),
            {},
            workflow="fl2va",
            ref2va_modality="video",
        )


def test_vision_compute_precision_resolves_each_declared_dtype_exactly() -> None:
    trt = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
    assert vision._resolve_compute_dtype("bf16", trt) == "bf16"
    assert vision._resolve_compute_dtype("fp16", trt) == "fp16"
    assert vision._resolve_compute_dtype("fp32", trt) == "fp32"
    with pytest.raises(ValueError, match="compute precision 'tf32'"):
        vision._resolve_compute_dtype("tf32", trt)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "qwen2_vl", "model_type='qwen3_vl'"),
        (("text_config", "dtype"), "float16", "dtype must be bfloat16"),
        (("vision_config", "patch_size"), 14, "'patch_size' must be 16"),
        (
            ("vision_config", "deepstack_visual_indexes"),
            [8, 16],
            "deepstack_visual_indexes",
        ),
    ],
)
def test_checkpoint_config_mismatches_fail_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = _checkpoint_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        vision.MiniMaxH3VisionConditionerSpec.from_checkpoint_config(config)


def test_checkpoint_key_contract_covers_every_official_visual_tensor() -> None:
    shapes = vision.expected_weight_shapes()
    assert tuple(shapes) == vision.checkpoint_keys()
    assert len(shapes) == 351
    assert shapes["model.visual.patch_embed.proj.weight"] == (1152, 3, 2, 16, 16)
    assert shapes["model.visual.pos_embed.weight"] == (2304, 1152)
    assert shapes["model.visual.blocks.26.attn.qkv.weight"] == (3456, 1152)
    assert shapes["model.visual.merger.norm.weight"] == (1152,)
    assert shapes["model.visual.deepstack_merger_list.2.norm.weight"] == (4608,)
    assert shapes["model.visual.deepstack_merger_list.2.linear_fc2.weight"] == (
        5120,
        4608,
    )


class _ShapeOnly:
    def __init__(self, shape: tuple[int, ...], dtype=np.float32):
        self.shape = shape
        self.dtype = np.dtype(dtype)


def _shape_only_weights(
    spec: vision.MiniMaxH3VisionConditionerSpec,
) -> dict[str, _ShapeOnly]:
    return {
        name: _ShapeOnly(shape, ml_dtypes.bfloat16)
        for name, shape in vision.expected_weight_shapes(spec).items()
    }


def test_weight_validation_accepts_bf16_without_materializing_checkpoint() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec()
    weights = _shape_only_weights(spec)
    weights["model.language_model.embed_tokens.weight"] = _ShapeOnly((151936, 5120))
    vision.validate_vision_weights(weights, spec)


def test_weight_validation_rejects_missing_extra_shape_and_lossy_dtype() -> None:
    spec = _toy_spec()

    weights = _shape_only_weights(spec)
    del weights["model.visual.pos_embed.weight"]
    with pytest.raises(ValueError, match="missing=.*pos_embed"):
        vision.validate_vision_weights(weights, spec)

    weights = _shape_only_weights(spec)
    weights["model.visual.unexpected.weight"] = _ShapeOnly((1,))
    with pytest.raises(ValueError, match="unexpected=.*unexpected"):
        vision.validate_vision_weights(weights, spec)

    weights = _shape_only_weights(spec)
    weights["model.visual.patch_embed.proj.bias"] = _ShapeOnly((3,))
    with pytest.raises(ValueError, match="must have shape"):
        vision.validate_vision_weights(weights, spec)

    weights = _shape_only_weights(spec)
    weights["model.visual.patch_embed.proj.bias"] = _ShapeOnly((8,), np.float16)
    with pytest.raises(ValueError, match="must be BF16 or FP32"):
        vision.validate_vision_weights(weights, spec)


def test_position_interpolation_preserves_processor_merge_group_order() -> None:
    spec = _toy_spec()
    rows, columns = vision._processor_merge_group_coordinates(spec)
    assert list(zip(rows.tolist(), columns.tolist())) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ]

    # A 2x2 learned table, repeated across the toy hidden width.  Resampling
    # to 2x4 keeps height exact and linearly stretches width with aligned ends.
    table = np.repeat(
        np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32),
        spec.hidden_size,
        axis=1,
    )
    interpolated = vision._interpolated_position_embeddings(table, spec)
    expected = np.asarray(
        [0.0, 1.0 / 3.0, 10.0, 10.375, 2.0 / 3.0, 1.0, 10.0 + 2.0 / 3.0, 11.0],
        dtype=ml_dtypes.bfloat16,
    ).astype(np.float32)
    np.testing.assert_array_equal(interpolated[:, 0].astype(np.float32), expected)
    assert interpolated.dtype == np.dtype(ml_dtypes.bfloat16)


def test_torch_linspace_fp32_matches_pinned_qwen_coordinates() -> None:
    image = vision._torch_linspace_fp32(0.0, 47.0, 128)
    video = vision._torch_linspace_fp32(0.0, 47.0, 48)
    assert hashlib.sha256(image.view(np.uint32).tobytes()).hexdigest() == (
        "f8b21ad59dfbb4036ec68e490d2c5c96db17db010f3d2b2c4efc4edab1d6ceeb"
    )
    assert hashlib.sha256(video.view(np.uint32).tobytes()).hexdigest() == (
        "77135df9eb160bde21ae2ace0f16da1ad544c3be39e09d8e080b4e593b7e0bd4"
    )
    expected_bits = {
        5: 0x3FECD9B4,
        10: 0x406CD9B4,
        20: 0x40ECD9B4,
        27: 0x411FDFC0,
        40: 0x416CD9B4,
        43: 0x417E9D3B,
        54: 0x419FDFC0,
        73: 0x41D82040,
        95: 0x420CA142,
    }
    bits = image.view(np.uint32)
    assert {index: int(bits[index]) for index in expected_bits} == expected_bits
    np.testing.assert_array_equal(video, np.arange(48, dtype=np.float32))


def test_h3_position_weights_match_pinned_aten_linspace_rounding() -> None:
    _, weights = vision._position_interpolation_indices_weights(
        vision.MiniMaxH3VisionConditionerSpec()
    )
    np.testing.assert_array_equal(
        weights[:8],
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.4337349534034729, 0.5662650465965271, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.4337349534034729, 0.5662650465965271, 0.0, 0.0],
                [0.8674699068069458, 0.1325300931930542, 0.0, 0.0],
                [0.3012049198150635, 0.6987950801849365, 0.0, 0.0],
                [0.8674699068069458, 0.1325300931930542, 0.0, 0.0],
                [0.3012049198150635, 0.6987950801849365, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_rope_tables_follow_the_same_merge_group_coordinates() -> None:
    spec = _toy_spec()
    cos_half, sin_half = vision._vision_rope_cos_sin(spec)
    assert cos_half.shape == sin_half.shape == (8, 2)
    rows, columns = vision._processor_merge_group_coordinates(spec)
    np.testing.assert_allclose(cos_half[:, 0], np.cos(rows.astype(np.float32)))
    np.testing.assert_allclose(sin_half[:, 0], np.sin(rows.astype(np.float32)))
    np.testing.assert_allclose(cos_half[:, 1], np.cos(columns.astype(np.float32)))
    np.testing.assert_allclose(sin_half[:, 1], np.sin(columns.astype(np.float32)))


def test_ref2va_runtime_positions_cover_max_image_and_video_blocks() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    image = vision.make_ref2va_position_bindings(128, 512, spec)
    assert image["position_indices"].shape == (65536, 4)
    assert image["position_weights"].shape == (65536, 4)
    assert image["vision_position_ids"].shape == (65536, 2)
    assert image["position_indices"].dtype == np.int32
    assert image["position_weights"].dtype == np.float32
    assert image["vision_position_ids"].dtype == np.int32
    np.testing.assert_array_equal(
        image["vision_position_ids"][:4],
        np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], np.int32),
    )

    assert vision._is_legal_video_grid(36, 116)
    video = vision.make_ref2va_position_bindings(36, 116, spec)
    assert video["position_indices"].shape == (4176, 4)
    assert video["vision_position_ids"].shape == (4176, 2)


def test_ref2va_binding_validator_accepts_actual_rows_and_rejects_drift() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    grid_h, grid_w = 48, 84
    bindings = vision.make_ref2va_position_bindings(grid_h, grid_w, spec)
    pixels = np.zeros(
        (grid_h * grid_w, spec.patch_vector_size),
        dtype=ml_dtypes.bfloat16,
    )
    assert (
        vision.validate_ref2va_vision_bindings(
            modality="video",
            grid_h=grid_h,
            grid_w=grid_w,
            pixel_values=pixels,
            spec=spec,
            **bindings,
        )
        == 1008
    )

    wrong_positions = {name: value.copy() for name, value in bindings.items()}
    wrong_positions["vision_position_ids"][0, 0] = 1
    with pytest.raises(ValueError, match="vision_position_ids"):
        vision.validate_ref2va_vision_bindings(
            modality="video",
            grid_h=grid_h,
            grid_w=grid_w,
            pixel_values=pixels,
            spec=spec,
            **wrong_positions,
        )

    with pytest.raises(ValueError, match="legal rounded 768p"):
        vision.validate_ref2va_vision_bindings(
            modality="video",
            grid_h=46,
            grid_w=52,
            pixel_values=np.zeros((1, 1), np.float32),
            position_indices=np.zeros((1, 4), np.int32),
            position_weights=np.zeros((1, 4), dtype=np.float32),
            vision_position_ids=np.zeros((1, 2), np.int32),
            spec=spec,
        )


class _FakeTensor:
    def __init__(self, label: str, dtype: object, shape: tuple[int, ...]):
        self.label = label
        self.dtype = dtype
        self.shape = shape
        self.name = ""
        self.dimension_names = {}

    def set_dimension_name(self, axis, name):
        self.dimension_names[axis] = name


class _FakeLayer:
    def __init__(self, output: _FakeTensor):
        self.output = output

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


class _FakeNetwork:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add_input(self, name, dtype, shape):
        tensor = _FakeTensor(name, dtype, shape)
        self.inputs.append(tensor)
        return tensor

    def add_elementwise(self, left, right, operation):
        del right, operation
        return _FakeLayer(_FakeTensor(f"{left.label}+position", left.dtype, left.shape))

    def mark_output(self, tensor):
        self.outputs.append(tensor)


class _FakeOps:
    @staticmethod
    def cast(network, tensor, dtype):
        del network
        return _FakeTensor(tensor.label, dtype, tensor.shape)

    @staticmethod
    def weight_constant(network, value):
        del network
        return _FakeTensor("position", "bf16", tuple(value.shape))


def test_mocked_graph_preserves_deepstack_capture_and_output_order(monkeypatch) -> None:
    spec = _toy_spec()
    fake_trt = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        ElementWiseOperation=SimpleNamespace(SUM="sum"),
    )
    merger_calls = []

    monkeypatch.setattr(
        vision,
        "_interpolated_position_embeddings",
        lambda table, current: np.zeros((current.num_patches, current.hidden_size), np.float32),
    )
    monkeypatch.setattr(
        vision,
        "_vision_rope_cos_sin",
        lambda current: (
            np.zeros((current.num_patches, current.head_dim // 2), np.float32),
            np.zeros((current.num_patches, current.head_dim // 2), np.float32),
        ),
    )
    monkeypatch.setattr(
        vision,
        "_patch_embedding",
        lambda network, pixels, weights, current, trt, op: _FakeTensor(
            "patch", trt.bfloat16, (current.num_patches, current.hidden_size)
        ),
    )
    monkeypatch.setattr(
        vision,
        "_patch_embedding_ref2va_image_plugin",
        lambda *_args, **_kwargs: pytest.fail("FL2VA must retain the TensorRT patch GEMM"),
    )

    def block(network, hidden, weights, index, cos, sin, current, trt, op):
        del network, weights, cos, sin, trt, op
        return _FakeTensor(
            f"block-{index}", hidden.dtype, (current.num_patches, current.hidden_size)
        )

    monkeypatch.setattr(vision, "_vision_block", block)

    def merger(network, hidden, weights, prefix, *, postshuffle_norm, spec, trt, op):
        del network, weights, trt, op
        merger_calls.append((prefix, postshuffle_norm, hidden.label))
        return _FakeTensor(prefix, "bf16", spec.output_shape)

    monkeypatch.setattr(vision, "_patch_merger", merger)
    network = _FakeNetwork()
    outputs = vision._assemble_vision_conditioner_graph(
        network,
        {"model.visual.pos_embed.weight": np.zeros((4, 4), np.float32)},
        spec,
        pixel_dtype="fp32",
        trt=fake_trt,
        op=_FakeOps,
    )
    assert [(item.label, item.dtype, item.shape) for item in network.inputs] == [
        ("pixel_values", "fp32", (8, 24))
    ]
    assert merger_calls == [
        ("model.visual.merger", False, "block-2"),
        ("model.visual.deepstack_merger_list.0", True, "block-0"),
        ("model.visual.deepstack_merger_list.1", True, "block-2"),
    ]
    assert [output.name for output in outputs] == [
        "image_features",
        "deepstack_features_0",
        "deepstack_features_1",
    ]
    assert network.outputs == list(outputs)
    assert all(output.shape == (2, 6) for output in outputs)


class _FakeProfile:
    def __init__(self):
        self.shapes = {}

    def set_shape(self, name, minimum, optimum, maximum):
        self.shapes[name] = (minimum, optimum, maximum)
        return True


class _FakeBuilder:
    def __init__(self):
        self.profile = _FakeProfile()

    def create_optimization_profile(self):
        return self.profile


class _FakeConfig:
    def __init__(self):
        self.profiles = []

    def add_optimization_profile(self, profile):
        self.profiles.append(profile)
        return len(self.profiles) - 1


def test_ref2va_dynamic_inputs_share_one_unpadded_patch_profile() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    trt = SimpleNamespace(int32="int32", bfloat16="bf16", float32="fp32")
    network = _FakeNetwork()
    inputs = vision._declare_ref2va_inputs(network, spec, "fp32", trt)
    assert inputs["pixel_values"].shape == (-1, 1536)
    assert inputs["position_indices"].shape == (-1, 4)
    assert inputs["position_weights"].shape == (-1, 4)
    assert inputs["position_weights"].dtype == "fp32"
    assert inputs["vision_position_ids"].shape == (-1, 2)
    assert all(tensor.dimension_names == {0: "vision_patch_rows"} for tensor in inputs.values())

    builder = _FakeBuilder()
    config = _FakeConfig()
    vision._add_ref2va_profile(builder, config, spec)
    assert builder.profile.shapes["pixel_values"] == (
        (2304, 1536),
        (4032, 1536),
        (65536, 1536),
    )
    assert builder.profile.shapes["position_indices"] == (
        (2304, 4),
        (4032, 4),
        (65536, 4),
    )
    assert len(config.profiles) == 1


def test_ref2va_position_and_attention_publication_boundaries_are_explicit() -> None:
    interpolation = inspect.getsource(vision._runtime_interpolated_positions)
    assert "blend = op.cast(network, blend, trt.bfloat16)" in interpolation
    assert interpolation.count("trt.ElementWiseOperation.SUM") == 3
    assert "return result" in interpolation

    attention = inspect.getsource(vision._vision_attention_dynamic)
    plugin_branch = attention.split("if attention_backend == _REF2VA_ATTENTION_BACKEND_PLUGIN:", 1)[
        1
    ].split("elif attention_backend == _REF2VA_ATTENTION_BACKEND_TRT:", 1)[0]
    trt_branch = attention.split("elif attention_backend == _REF2VA_ATTENTION_BACKEND_TRT:", 1)[1]
    assert "_add_ref2va_image_attention_plugin" in plugin_branch
    assert "network.add_attention" not in plugin_branch
    assert "math.sqrt" not in plugin_branch
    assert "trt.bfloat16" in plugin_branch
    assert "_rows_to_heads_dynamic" not in plugin_branch
    assert "_heads_to_rows_dynamic" not in plugin_branch

    attention_call = trt_branch.index("network.add_attention")
    for name in ("k", "v"):
        assert (
            trt_branch.index(f"{name} = op.cast(network, {name}, attention_dtype)") < attention_call
        )
    assert trt_branch.index("q = op.cast(network, q, q_scale_dtype)") < attention_call
    assert trt_branch.index("q = op.cast(network, q, attention_dtype)") < attention_call
    assert "math.sqrt(spec.head_dim)" in trt_branch

    builder = inspect.getsource(vision.build_vision_conditioner_engine)
    load = builder.index("_load_ref2va_native_plugin")
    create_builder = builder.index("builder = trt.Builder")
    assert load < create_builder
    assert 'if ref2va_modality == "image"' in builder
    assert "_REF2VA_PATCH_BACKEND_PLUGIN" in builder
    assert "_REF2VA_LINEAR_BACKEND_PLUGIN" in builder
    assert "_REF2VA_NORM_BACKEND_PLUGIN" in builder

    ref2va_graph = inspect.getsource(vision._assemble_ref2va_vision_graph)
    assert "_patch_embedding_ref2va_image_plugin" in ref2va_graph
    assert "if patch_backend == _REF2VA_PATCH_BACKEND_PLUGIN" in ref2va_graph
    assert "linear_backend=linear_backend" in ref2va_graph
    assert inspect.getsource(vision._vision_attention_dynamic).count("_linear_ref2va(") == 2
    assert inspect.getsource(vision._vision_block_dynamic).count("_linear_ref2va(") == 2
    assert inspect.getsource(vision._patch_merger_dynamic).count("_linear_ref2va(") == 2
    assert inspect.getsource(vision._vision_block_dynamic).count("_layer_norm_ref2va(") == 2
    assert inspect.getsource(vision._patch_merger_dynamic).count("_layer_norm_ref2va(") == 2


def test_ref2va_image_patch_embed_uses_exact_bf16_v3_inputs_without_gemm(monkeypatch) -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    prefix = "model.visual.patch_embed.proj"
    weights = {
        f"{prefix}.weight": np.zeros(
            (1152, 3, 2, 16, 16),
            dtype=ml_dtypes.bfloat16,
        ),
        f"{prefix}.bias": np.zeros((1152,), dtype=ml_dtypes.bfloat16),
    }
    constants = []

    class Ops:
        @staticmethod
        def weight_constant(network, value):
            del network
            constants.append(value)
            return _FakeTensor("constant", value.dtype, tuple(value.shape))

        @staticmethod
        def cast(network, tensor, dtype):
            del network
            return _FakeTensor(tensor.label, dtype, tensor.shape)

        @staticmethod
        def linear(*_args, **_kwargs):
            pytest.fail("Ref2VA image patch plugin path must not add a GEMM")

    plugin_calls = []

    def add_plugin(network, pixel, weight, bias, trt, *, name):
        del network, trt
        plugin_calls.append((pixel, weight, bias, name))
        return _FakeTensor("patch-output", "bf16", (-1, 1152))

    monkeypatch.setattr(vision, "_add_ref2va_image_patch_embed_plugin", add_plugin)
    pixels = _FakeTensor("pixels", "bf16", (-1, 1536))
    result = vision._patch_embedding_ref2va_image_plugin(
        object(),
        pixels,
        weights,
        spec,
        SimpleNamespace(bfloat16="bf16"),
        Ops,
    )

    assert [tuple(value.shape) for value in constants] == [
        (1152, 3, 2, 16, 16),
        (1152,),
    ]
    assert all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in constants)
    assert len(plugin_calls) == 1
    pixel, weight, bias, name = plugin_calls[0]
    assert (pixel.dtype, pixel.shape) == ("bf16", (-1, 1536))
    assert (weight.dtype, weight.shape) == ("bf16", (1152, 3, 2, 16, 16))
    assert (bias.dtype, bias.shape) == ("bf16", (1152,))
    assert name == "model.visual.patch_embed.proj.hf_conv3d"
    assert (result.dtype, result.shape) == ("bf16", (-1, 1152))


def test_ref2va_image_linear_uses_exact_bf16_v3_inputs_without_gemm(monkeypatch) -> None:
    weight_value = np.arange(32, dtype=np.float32).reshape(8, 4)
    bias_value = np.arange(8, dtype=np.float32)
    constants = []

    class Ops:
        @staticmethod
        def weight_constant(network, value):
            del network
            constants.append(value)
            return _FakeTensor("constant", value.dtype, tuple(value.shape))

        @staticmethod
        def cast(network, tensor, dtype):
            del network
            return _FakeTensor(tensor.label, dtype, tensor.shape)

        @staticmethod
        def linear(*_args, **_kwargs):
            pytest.fail("Ref2VA image Linear plugin path must not add a TensorRT GEMM")

    plugin_calls = []

    def add_plugin(network, tensor, weight, bias, trt, *, name):
        del network, trt
        plugin_calls.append((tensor, weight, bias, name))
        return _FakeTensor("linear-output", "bf16", (-1, 8))

    monkeypatch.setattr(vision, "_add_ref2va_image_linear_plugin", add_plugin)
    result = vision._linear_ref2va(
        object(),
        _FakeTensor("hidden", "bf16", (-1, 4)),
        weight_value,
        bias_value,
        SimpleNamespace(bfloat16="bf16"),
        Ops,
        linear_backend=vision._REF2VA_LINEAR_BACKEND_PLUGIN,
        name="model.visual.blocks.0.attn.qkv.hf_linear",
    )

    assert [tuple(value.shape) for value in constants] == [(8, 4), (8,)]
    assert all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in constants)
    assert len(plugin_calls) == 1
    tensor, weight, bias, name = plugin_calls[0]
    assert (tensor.dtype, tensor.shape) == ("bf16", (-1, 4))
    assert (weight.dtype, weight.shape) == ("bf16", (8, 4))
    assert (bias.dtype, bias.shape) == ("bf16", (8,))
    assert name == "model.visual.blocks.0.attn.qkv.hf_linear"
    assert (result.dtype, result.shape) == ("bf16", (-1, 8))


def test_ref2va_image_layer_norm_uses_exact_bf16_v3_inputs(monkeypatch) -> None:
    constants = []

    class Ops:
        @staticmethod
        def weight_constant(network, value):
            del network
            constants.append(value)
            return _FakeTensor("constant", value.dtype, tuple(value.shape))

        @staticmethod
        def cast(network, tensor, dtype):
            del network
            return _FakeTensor(tensor.label, dtype, tensor.shape)

    plugin_calls = []

    def add_plugin(network, tensor, weight, bias, trt, *, name):
        del network, trt
        plugin_calls.append((tensor, weight, bias, name))
        return _FakeTensor("norm-output", "bf16", (-1, 4))

    monkeypatch.setattr(vision, "_add_ref2va_image_layer_norm_plugin", add_plugin)
    result = vision._layer_norm_ref2va(
        object(),
        _FakeTensor("hidden", "bf16", (-1, 4)),
        np.ones((4,), np.float32),
        np.zeros((4,), np.float32),
        4,
        1.0e-6,
        SimpleNamespace(bfloat16="bf16"),
        Ops,
        norm_backend=vision._REF2VA_NORM_BACKEND_PLUGIN,
        name="model.visual.blocks.0.norm1.hf_layer_norm",
    )

    assert [tuple(value.shape) for value in constants] == [(4,), (4,)]
    assert all(value.dtype == np.dtype(ml_dtypes.bfloat16) for value in constants)
    assert len(plugin_calls) == 1
    tensor, weight, bias, name = plugin_calls[0]
    assert (tensor.dtype, tensor.shape) == ("bf16", (-1, 4))
    assert (weight.dtype, weight.shape) == ("bf16", (4,))
    assert (bias.dtype, bias.shape) == ("bf16", (4,))
    assert name == "model.visual.blocks.0.norm1.hf_layer_norm"
    assert (result.dtype, result.shape) == ("bf16", (-1, 4))


def test_ref2va_video_and_fl_keep_tensor_rt_linear_layers(monkeypatch) -> None:
    calls = []

    class Ops:
        @staticmethod
        def linear(network, tensor, weight, bias, *, compute_dtype):
            del network
            calls.append((tensor, weight, bias, compute_dtype))
            return _FakeTensor("trt-linear", compute_dtype, (-1, 8))

    monkeypatch.setattr(
        vision,
        "_add_ref2va_image_linear_plugin",
        lambda *_args, **_kwargs: pytest.fail("video must not add the image Linear plugin"),
    )
    output = vision._linear_ref2va(
        object(),
        _FakeTensor("hidden", "bf16", (-1, 4)),
        object(),
        object(),
        SimpleNamespace(bfloat16="bf16"),
        Ops,
        linear_backend=vision._REF2VA_LINEAR_BACKEND_TRT,
        name="unused-on-video",
    )
    assert len(calls) == 1
    assert calls[0][3] == "bf16"
    assert output.label == "trt-linear"

    for fixed_helper in (vision._vision_attention, vision._vision_block, vision._patch_merger):
        assert "_linear_ref2va" not in inspect.getsource(fixed_helper)


def test_ref2va_video_and_fl_keep_tensor_rt_layer_norm(monkeypatch) -> None:
    calls = []

    def native_norm(network, tensor, weight, bias, width, eps, trt, op):
        calls.append((network, tensor, weight, bias, width, eps, trt, op))
        return _FakeTensor("trt-norm", tensor.dtype, tensor.shape)

    monkeypatch.setattr(vision, "_layer_norm", native_norm)
    monkeypatch.setattr(
        vision,
        "_add_ref2va_image_layer_norm_plugin",
        lambda *_args, **_kwargs: pytest.fail("video must not add the image LayerNorm plugin"),
    )
    result = vision._layer_norm_ref2va(
        object(),
        _FakeTensor("hidden", "bf16", (-1, 4)),
        object(),
        object(),
        4,
        1.0e-6,
        SimpleNamespace(bfloat16="bf16"),
        object(),
        norm_backend=vision._REF2VA_NORM_BACKEND_TRT,
        name="unused-on-video",
    )
    assert len(calls) == 1
    assert result.label == "trt-norm"

    for fixed_helper in (vision._vision_block, vision._patch_merger):
        assert "_layer_norm_ref2va" not in inspect.getsource(fixed_helper)


def _run_mocked_dynamic_attention(monkeypatch, backend: str):
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    prefix = "model.visual.blocks.0"
    trt = SimpleNamespace(
        bfloat16="bf16",
        AttentionNormalizationOp=SimpleNamespace(SOFTMAX="softmax"),
        ElementWiseOperation=SimpleNamespace(PROD="prod"),
    )

    class Network:
        def __init__(self):
            self.attention_calls = []
            self.products = []

        def add_elementwise(self, left, right, operation):
            self.products.append((left, right, operation))
            return _FakeLayer(_FakeTensor("scaled-q", left.dtype, left.shape))

        def add_attention(self, q, k, v, normalization, causal):
            self.attention_calls.append((q, k, v, normalization, causal))
            return _FakeLayer(_FakeTensor("trt-context", q.dtype, q.shape))

    class Ops:
        def __init__(self):
            self.casts = []
            self.linear_calls = []
            self.constants = []

        def linear(self, network, tensor, weight, bias, *, compute_dtype):
            del network, weight, bias
            self.linear_calls.append((tensor, compute_dtype))
            if len(self.linear_calls) == 1:
                return _FakeTensor("qkv", compute_dtype, (-1, 3 * spec.hidden_size))
            return _FakeTensor("projection", compute_dtype, (-1, spec.hidden_size))

        def cast(self, network, tensor, dtype):
            del network
            self.casts.append((tensor.label, tensor.dtype, dtype))
            return _FakeTensor(tensor.label, dtype, tensor.shape)

        def constant(self, network, value):
            del network
            self.constants.append(np.asarray(value))
            return _FakeTensor("scale", "fp32", tuple(value.shape))

    network = Network()
    ops = Ops()
    head_conversions = []
    row_conversions = []
    monkeypatch.setattr(
        vision,
        "_dynamic_column_slice",
        lambda network, tensor, start, width, op: _FakeTensor(
            {0: "q", spec.hidden_size: "k", 2 * spec.hidden_size: "v"}[start],
            "bf16",
            (-1, width),
        ),
    )
    monkeypatch.setattr(
        vision,
        "_apply_vision_rope_dynamic",
        lambda network, tensor, cosine, sine, heads, current, trt, op: _FakeTensor(
            f"{tensor.label}-rope", tensor.dtype, tensor.shape
        ),
    )

    def rows_to_heads(network, tensor, heads, head_dim, trt):
        del network, trt
        head_conversions.append(tensor.label)
        return _FakeTensor(tensor.label, tensor.dtype, (1, heads, -1, head_dim))

    def heads_to_rows(network, tensor, width, trt):
        del network, trt
        row_conversions.append(tensor.label)
        return _FakeTensor(tensor.label, tensor.dtype, (-1, width))

    monkeypatch.setattr(vision, "_rows_to_heads_dynamic", rows_to_heads)
    monkeypatch.setattr(vision, "_heads_to_rows_dynamic", heads_to_rows)
    weights = {
        f"{prefix}.attn.qkv.weight": object(),
        f"{prefix}.attn.qkv.bias": object(),
        f"{prefix}.attn.proj.weight": object(),
        f"{prefix}.attn.proj.bias": object(),
    }
    result = vision._vision_attention_dynamic(
        network,
        _FakeTensor("hidden", "bf16", (-1, spec.hidden_size)),
        weights,
        prefix,
        object(),
        object(),
        spec,
        backend,
        vision._REF2VA_LINEAR_BACKEND_TRT,
        "fp16",
        "fp16",
        trt,
        ops,
    )
    return network, ops, result, head_conversions, row_conversions


def test_ref2va_image_attention_uses_bf16_plugin_without_prescaling(monkeypatch) -> None:
    plugin_calls = []

    def add_plugin(network, q, k, v, trt, *, name):
        del network, trt
        plugin_calls.append((q, k, v, name))
        return _FakeTensor("plugin-context", "bf16", q.shape)

    monkeypatch.setattr(vision, "_add_ref2va_image_attention_plugin", add_plugin)
    network, ops, result, head_conversions, row_conversions = _run_mocked_dynamic_attention(
        monkeypatch, vision._REF2VA_ATTENTION_BACKEND_PLUGIN
    )

    assert len(plugin_calls) == 1
    q, k, v, name = plugin_calls[0]
    assert name == "model.visual.blocks.0.attn.hf_sdpa"
    assert [(tensor.dtype, tensor.shape) for tensor in (q, k, v)] == [
        ("bf16", (-1, 1152)),
        ("bf16", (-1, 1152)),
        ("bf16", (-1, 1152)),
    ]
    assert head_conversions == []
    assert row_conversions == []
    assert network.attention_calls == []
    assert network.products == []
    assert ops.constants == []
    assert result.dtype == "bf16"


def test_ref2va_video_attention_retains_fp16_iattention_and_q_scale(monkeypatch) -> None:
    monkeypatch.setattr(
        vision,
        "_add_ref2va_image_attention_plugin",
        lambda *args, **kwargs: pytest.fail("video path must not add the image attention plugin"),
    )
    network, ops, result, head_conversions, row_conversions = _run_mocked_dynamic_attention(
        monkeypatch, vision._REF2VA_ATTENTION_BACKEND_TRT
    )

    assert len(network.attention_calls) == 1
    q, k, v, normalization, causal = network.attention_calls[0]
    assert [(tensor.dtype, tensor.shape) for tensor in (q, k, v)] == [
        ("fp16", (1, 16, -1, 72)),
        ("fp16", (1, 16, -1, 72)),
        ("fp16", (1, 16, -1, 72)),
    ]
    assert head_conversions == ["q-rope", "k-rope", "v"]
    assert row_conversions == ["trt-context"]
    assert normalization == "softmax"
    assert causal is False
    assert len(network.products) == 1
    assert network.products[0][2] == "prod"
    assert len(ops.constants) == 1
    np.testing.assert_array_equal(
        ops.constants[0],
        np.full((1, 1, 1, 1), 1.0 / np.sqrt(72), np.float32),
    )
    assert result.dtype == "bf16"


def test_ref2va_image_network_requires_202_exact_named_v3_plugins() -> None:
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow("ref2va")
    layer_types = SimpleNamespace(
        PLUGIN="plugin",
        PLUGIN_V2="plugin_v2",
        PLUGIN_V3="plugin_v3",
        ATTENTION_INPUT="attention_input",
        ATTENTION_OUTPUT="attention_output",
        MATRIX_MULTIPLY="matrix_multiply",
        NORMALIZATION="normalization",
        REDUCE="reduce",
        DIST_COLLECTIVE="dist_collective",
    )
    trt = SimpleNamespace(LayerType=layer_types)

    class Layer:
        def __init__(self, kind, name, metadata=""):
            self.type = kind
            self.name = name
            self.metadata = metadata

    class Network:
        def __init__(self, layers):
            self.layers = layers
            self.num_layers = len(layers)

        def get_layer(self, index):
            return self.layers[index]

    linear_names = [
        *(
            f"model.visual.blocks.{index}.{suffix}.hf_linear"
            for index in range(spec.depth)
            for suffix in (
                "attn.qkv",
                "attn.proj",
                "mlp.linear_fc1",
                "mlp.linear_fc2",
            )
        ),
        *(
            f"{prefix}.{suffix}.hf_linear"
            for prefix in (
                "model.visual.merger",
                *(
                    f"model.visual.deepstack_merger_list.{index}"
                    for index in range(len(spec.deepstack_visual_indexes))
                ),
            )
            for suffix in ("linear_fc1", "linear_fc2")
        ),
    ]
    assert len(linear_names) == 116
    norm_names = [
        *(
            f"model.visual.blocks.{index}.{suffix}.hf_layer_norm"
            for index in range(spec.depth)
            for suffix in ("norm1", "norm2")
        ),
        "model.visual.merger.norm.hf_layer_norm",
        *(
            f"model.visual.deepstack_merger_list.{index}.norm.hf_layer_norm"
            for index in range(len(spec.deepstack_visual_indexes))
        ),
    ]
    assert len(norm_names) == 58
    plugins = [
        Layer(layer_types.PLUGIN_V3, "model.visual.patch_embed.proj.hf_conv3d"),
        *[
            Layer(layer_types.PLUGIN_V3, f"model.visual.blocks.{index}.attn.hf_sdpa")
            for index in range(spec.depth)
        ],
        *(Layer(layer_types.PLUGIN_V3, name) for name in linear_names),
        *(Layer(layer_types.PLUGIN_V3, name) for name in norm_names),
    ]
    for layer in plugins:
        creator = (
            "MiniMaxH3PatchEmbed"
            if layer.name.endswith(".hf_conv3d")
            else "MiniMaxH3VisionAttention"
            if layer.name.endswith(".hf_sdpa")
            else "MiniMaxH3Linear"
            if layer.name.endswith(".hf_linear")
            else "MiniMaxH3LayerNorm"
        )
        layer.metadata = f"trtmc.native_op={creator};source={layer.name}"
    network = Network([Layer("elementwise", "position-sum"), *plugins])
    assert vision._validate_ref2va_plugin_network(network, spec, trt) == {
        "attention_input": 0,
        "attention_output": 0,
        "plugin_v3": 202,
        "dist_collective": 0,
    }

    plugins[-1] = Layer(layer_types.PLUGIN_V2, "model.visual.blocks.26.attn.hf_sdpa")
    with pytest.raises(RuntimeError, match="plugin_v2"):
        vision._validate_ref2va_plugin_network(Network(plugins), spec, trt)
    final_norm_name = "model.visual.deepstack_merger_list.2.norm.hf_layer_norm"
    plugins[-1] = Layer(
        layer_types.PLUGIN_V3,
        final_norm_name,
        f"trtmc.native_op=MiniMaxH3LayerNorm;source={final_norm_name}",
    )

    for forbidden in (
        layer_types.MATRIX_MULTIPLY,
        layer_types.NORMALIZATION,
        layer_types.REDUCE,
    ):
        with pytest.raises(RuntimeError, match=forbidden):
            vision._validate_ref2va_plugin_network(
                Network(
                    [
                        *plugins[:-1],
                        Layer(
                            layer_types.PLUGIN_V3,
                            final_norm_name,
                            f"trtmc.native_op=MiniMaxH3LayerNorm;source={final_norm_name}",
                        ),
                        Layer(forbidden, f"unexpected-{forbidden}"),
                    ]
                ),
                spec,
                trt,
            )

    plugins[-1].metadata = f"trtmc.native_op=MiniMaxH3Linear;source={final_norm_name}"
    with pytest.raises(RuntimeError, match="plugin_metadata"):
        vision._validate_ref2va_plugin_network(Network(plugins), spec, trt)


def test_mocked_ref2va_graph_keeps_dynamic_rows_and_all_deepstack_outputs(
    monkeypatch,
) -> None:
    spec = _toy_ref2va_spec()
    trt = SimpleNamespace(
        int32="int32",
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(SUM="sum"),
    )
    network = _FakeNetwork()
    inputs = {
        "pixel_values": _FakeTensor("pixel_values", "fp32", (-1, 24)),
        "position_indices": _FakeTensor("position_indices", "int32", (-1, 4)),
        "position_weights": _FakeTensor("position_weights", "fp32", (-1, 4)),
        "vision_position_ids": _FakeTensor("vision_position_ids", "int32", (-1, 2)),
    }
    merger_calls = []
    monkeypatch.setattr(
        vision,
        "_patch_embedding",
        lambda network, pixels, weights, current, trt, op: _FakeTensor(
            "patch", "bf16", (-1, current.hidden_size)
        ),
    )
    monkeypatch.setattr(
        vision,
        "_patch_embedding_ref2va_image_plugin",
        lambda *_args, **_kwargs: pytest.fail("Ref2VA video must retain the TensorRT patch GEMM"),
    )
    monkeypatch.setattr(
        vision,
        "_runtime_interpolated_positions",
        lambda *args: _FakeTensor("positions", "bf16", (-1, spec.hidden_size)),
    )
    monkeypatch.setattr(
        vision,
        "_runtime_vision_rope_tables",
        lambda *args: (object(), object()),
    )

    def block(
        network,
        hidden,
        weights,
        index,
        cosine,
        sine,
        current,
        attention_backend,
        linear_backend,
        norm_backend,
        attention_dtype,
        q_scale_dtype,
        trt,
        op,
    ):
        del network, weights, cosine, sine, attention_dtype, q_scale_dtype, trt, op
        assert attention_backend == vision._REF2VA_ATTENTION_BACKEND_TRT
        assert linear_backend == vision._REF2VA_LINEAR_BACKEND_TRT
        assert norm_backend == vision._REF2VA_NORM_BACKEND_TRT
        return _FakeTensor(f"block-{index}", hidden.dtype, (-1, current.hidden_size))

    monkeypatch.setattr(vision, "_vision_block_dynamic", block)

    def merger(
        network,
        hidden,
        weights,
        prefix,
        *,
        postshuffle_norm,
        spec,
        linear_backend,
        norm_backend,
        trt,
        op,
    ):
        del network, weights, trt, op
        assert linear_backend == vision._REF2VA_LINEAR_BACKEND_TRT
        assert norm_backend == vision._REF2VA_NORM_BACKEND_TRT
        merger_calls.append((prefix, postshuffle_norm, hidden.label))
        return _FakeTensor(prefix, "bf16", (-1, spec.out_hidden_size))

    monkeypatch.setattr(vision, "_patch_merger_dynamic", merger)
    weights = {"model.visual.pos_embed.weight": _ShapeOnly((4, 8))}
    outputs = vision._assemble_ref2va_vision_graph(network, weights, spec, inputs, trt, _FakeOps)
    assert merger_calls == [
        ("model.visual.merger", False, "block-2"),
        ("model.visual.deepstack_merger_list.0", True, "block-0"),
        ("model.visual.deepstack_merger_list.1", True, "block-2"),
    ]
    assert [output.name for output in outputs] == [
        "image_features",
        "deepstack_features_0",
        "deepstack_features_1",
    ]
    assert all(output.shape == (-1, 6) for output in outputs)


@pytest.mark.gpu
def test_tiny_dynamic_ref2va_vision_graph_serializes_when_tensorrt_is_available() -> None:
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.families.minimax_h3 import graph_ops as op

    trt = trt_compat.get_trt()
    try:
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    except Exception as error:
        pytest.skip(f"TensorRT builder initialization is unavailable: {error}")
    if builder is None:
        pytest.skip("TensorRT builder initialization returned null")
    # Keep production's 16 heads x 72 channels so TensorRT exercises the same
    # dedicated attention kernel while rows/layers stay synthetic-small.
    spec = vision.MiniMaxH3VisionConditionerSpec.for_workflow(
        "ref2va",
        image_height=4,
        image_width=8,
        hidden_size=1152,
        intermediate_size=256,
        num_heads=16,
        depth=3,
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=128,
        deepstack_visual_indexes=(0, 2),
        min_patches=4,
        opt_patches=8,
        max_patches=16,
    )
    rng = np.random.default_rng(20260824)
    weights = {
        name: (
            np.ones(shape, dtype=ml_dtypes.bfloat16)
            if name.endswith("norm.weight")
            else rng.normal(0.0, 0.02, shape).astype(ml_dtypes.bfloat16)
        )
        for name, shape in vision.expected_weight_shapes(spec).items()
    }
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    op.configure_builder(config)
    op.configure_workspace(config, 1 << 30, default_bytes=1 << 30)
    inputs = vision._declare_ref2va_inputs(network, spec, "fp32", trt)
    vision._add_ref2va_profile(builder, config, spec)
    try:
        vision._assemble_ref2va_vision_graph(network, weights, spec, inputs, trt, op)
        op.validate_native_network(network, expected_attentions=spec.depth, label="tiny vision")
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
    assert plan is not None
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_shape("pixel_values") == (-1, spec.patch_vector_size)
    assert engine.get_tensor_shape("position_weights") == (-1, 4)
    assert engine.get_tensor_dtype("position_weights") == trt.float32


def test_pixel_binding_rejects_non_hf_dtypes() -> None:
    fake_trt = SimpleNamespace(float32="fp32", bfloat16="bf16")
    assert vision._resolve_pixel_dtype("fp32", fake_trt) == "fp32"
    assert vision._resolve_pixel_dtype("bf16", fake_trt) == "bf16"
    with pytest.raises(ValueError, match="'fp32' or 'bf16'"):
        vision._resolve_pixel_dtype("fp16", fake_trt)
