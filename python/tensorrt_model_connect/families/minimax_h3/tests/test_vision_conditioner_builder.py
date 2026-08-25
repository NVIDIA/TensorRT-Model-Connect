# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        [0.0, 1.0 / 3.0, 10.0, 10.3125, 2.0 / 3.0, 1.0, 10.0 + 2.0 / 3.0, 11.0],
        dtype=ml_dtypes.bfloat16,
    ).astype(np.float32)
    np.testing.assert_array_equal(interpolated[:, 0].astype(np.float32), expected)
    assert interpolated.dtype == np.dtype(ml_dtypes.bfloat16)


def test_h3_position_weights_match_pinned_multiply_then_divide_rounding() -> None:
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
                [0.3012048006057739, 0.6987951993942261, 0.0, 0.0],
                [0.8674699068069458, 0.1325300931930542, 0.0, 0.0],
                [0.3012048006057739, 0.6987951993942261, 0.0, 0.0],
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
        "_runtime_interpolated_positions",
        lambda *args: _FakeTensor("positions", "bf16", (-1, spec.hidden_size)),
    )
    monkeypatch.setattr(
        vision,
        "_runtime_vision_rope_tables",
        lambda *args: (object(), object()),
    )

    def block(network, hidden, weights, index, cosine, sine, current, trt, op):
        del network, weights, cosine, sine, trt, op
        return _FakeTensor(f"block-{index}", hidden.dtype, (-1, current.hidden_size))

    monkeypatch.setattr(vision, "_vision_block_dynamic", block)

    def merger(network, hidden, weights, prefix, *, postshuffle_norm, spec, trt, op):
        del network, weights, trt, op
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
    # dedicated BF16 attention kernel while rows/layers stay synthetic-small.
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
