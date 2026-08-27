# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3 import language_conditioner_builder as language


def _checkpoint_config() -> dict:
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 151655,
        "model_type": "qwen3_vl",
        "video_token_id": 151656,
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 64,
            "num_hidden_layers": 64,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1.0e-6,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5_000_000,
            "vocab_size": 151936,
        },
    }


def _toy_spec(*, output_layers: int = 4) -> language.MiniMaxH3LanguageConditionerSpec:
    return language.MiniMaxH3LanguageConditionerSpec(
        hidden_size=8,
        intermediate_size=12,
        vocab_size=16,
        available_layers=4,
        output_layers=output_layers,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        mrope_section=(4, 2, 2),
        min_rows=1,
        opt_rows=4,
        max_rows=16,
        vision_rows_per_keyframe=2,
        max_keyframes=2,
        image_token_id=15,
        video_token_id=12,
        vision_start_token_id=13,
        vision_end_token_id=14,
    )


def _toy_ref2va_spec() -> language.MiniMaxH3LanguageConditionerSpec:
    return replace(
        _toy_spec(),
        workflow="ref2va",
        opt_rows=8,
        max_rows=64,
        max_reference_images=2,
        max_reference_videos=2,
        max_references=3,
        max_video_runs=4,
        max_image_run_rows=6,
        max_video_run_rows=4,
    )


def test_h3_dynamic_language_contract_is_exact() -> None:
    spec = language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(_checkpoint_config())
    assert (spec.min_rows, spec.opt_rows, spec.max_rows) == (1, 537, 4096)
    assert (spec.available_layers, spec.output_layers) == (64, 50)
    assert (spec.attention_size, spec.kv_attention_size) == (8192, 1024)
    assert spec.mrope_section == (24, 20, 20)
    assert spec.mrope_interleaved is True
    assert spec.allowed_vision_rows == (0, 1008, 2016)


def test_ref2va_profile_reuses_graph_weights_with_full_context_envelope() -> None:
    config = _checkpoint_config()
    fl2va = language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(config)
    ref2va = language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(
        config, workflow="ref2va"
    )
    assert (fl2va.min_rows, fl2va.opt_rows, fl2va.max_rows) == (1, 537, 4096)
    assert (ref2va.min_rows, ref2va.opt_rows, ref2va.max_rows) == (
        1,
        8192,
        262144,
    )
    assert ref2va.workflow == "ref2va"
    assert ref2va.allowed_vision_rows == ()
    assert (ref2va.max_reference_images, ref2va.max_reference_videos) == (9, 3)
    assert (ref2va.max_image_run_rows, ref2va.max_video_run_rows) == (16384, 1044)
    assert ref2va.max_video_runs == 17
    assert language.expected_weight_shapes(fl2va) == language.expected_weight_shapes(ref2va)
    builder = _FakeBuilder()
    trt_config = _FakeConfig()
    language._add_optimization_profile(builder, trt_config, ref2va)
    assert builder.profile.shapes["input_ids"] == (
        (1,),
        (8192,),
        (262144,),
    )
    assert builder.profile.shapes["vision_embeddings"] == (
        (1, 5120),
        (8192, 5120),
        (262144, 5120),
    )
    assert "attention_mask" not in builder.profile.shapes

    too_short = _checkpoint_config()
    too_short["text_config"]["max_position_embeddings"] = 4096
    language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(too_short)
    with pytest.raises(ValueError, match="ref2va profile maximum 262144"):
        language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(
            too_short, workflow="ref2va"
        )


def test_public_builder_routes_the_explicit_ref2va_workflow(monkeypatch) -> None:
    observed = {}
    monkeypatch.setattr(language, "validate_conditioner_weights", lambda weights, spec: None)

    def build(weights, spec, **kwargs):
        observed.update(
            workflow=spec.workflow, profile=(spec.min_rows, spec.opt_rows, spec.max_rows)
        )
        return b"ref2va-plan"

    monkeypatch.setattr(language, "_build_language_conditioner_engine", build)
    assert (
        language.build_language_conditioner_engine(_checkpoint_config(), {}, workflow="ref2va")
        == b"ref2va-plan"
    )
    assert observed == {"workflow": "ref2va", "profile": (1, 8192, 262144)}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "qwen2_vl", "model_type='qwen3_vl'"),
        (("text_config", "dtype"), "float16", "dtype.*bfloat16"),
        (("text_config", "num_hidden_layers"), 50, "num_hidden_layers.*64"),
        (
            ("text_config", "rope_scaling", "mrope_interleaved"),
            False,
            "must be interleaved",
        ),
        (
            ("text_config", "rope_scaling", "mrope_section"),
            [32, 16, 16],
            "mrope_section",
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
        language.MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(config)


def test_weight_contract_stops_at_hidden_state_50() -> None:
    shapes = language.expected_weight_shapes()
    assert tuple(shapes) == language.checkpoint_keys()
    assert len(shapes) == 1 + 50 * 11 == 551
    assert shapes["model.language_model.embed_tokens.weight"] == (151936, 5120)
    assert shapes["model.language_model.layers.49.self_attn.q_proj.weight"] == (
        8192,
        5120,
    )
    assert shapes["model.language_model.layers.49.self_attn.k_proj.weight"] == (
        1024,
        5120,
    )
    assert "model.language_model.layers.50.input_layernorm.weight" not in shapes
    assert "model.language_model.norm.weight" not in shapes
    assert not any("lm_head" in name for name in shapes)


class _ShapeOnly:
    def __init__(self, shape: tuple[int, ...], dtype=ml_dtypes.bfloat16):
        self.shape = shape
        self.dtype = np.dtype(dtype)


def _shape_only_weights(
    spec: language.MiniMaxH3LanguageConditionerSpec,
) -> dict[str, _ShapeOnly]:
    return {
        name: _ShapeOnly(shape) for name, shape in language.expected_weight_shapes(spec).items()
    }


def test_weight_validation_rejects_missing_remaining_layer_and_lossy_dtype() -> None:
    spec = _toy_spec(output_layers=3)
    weights = _shape_only_weights(spec)
    language.validate_conditioner_weights(weights, spec)

    missing = dict(weights)
    del missing["model.language_model.layers.2.mlp.down_proj.weight"]
    with pytest.raises(ValueError, match="missing=.*down_proj"):
        language.validate_conditioner_weights(missing, spec)

    remaining = dict(weights)
    remaining["model.language_model.layers.3.input_layernorm.weight"] = _ShapeOnly((8,))
    with pytest.raises(ValueError, match="unexpected=.*layers.3"):
        language.validate_conditioner_weights(remaining, spec)

    lossy = dict(weights)
    lossy["model.language_model.layers.0.self_attn.q_norm.weight"] = _ShapeOnly(
        (spec.head_dim,), np.float16
    )
    with pytest.raises(ValueError, match="must be BF16 or FP32"):
        language.validate_conditioner_weights(lossy, spec)


def test_interleaved_mrope_axis_map_has_qwen3_temporal_tail() -> None:
    axes = language._mrope_frequency_axis_map((24, 20, 20), 128, interleaved=True)
    assert axes.shape == (64,)
    assert [int(np.count_nonzero(axes == axis)) for axis in range(3)] == [24, 20, 20]
    np.testing.assert_array_equal(
        axes[:12], np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], np.int32)
    )
    np.testing.assert_array_equal(axes[60:], np.zeros((4,), np.int32))


def test_rope_tables_cover_dynamic_profile_without_expanding_bf16() -> None:
    spec = _toy_spec()
    cosine, sine = language._make_rope_tables(spec)
    assert cosine.shape == sine.shape == (16, 8)
    assert cosine.dtype == sine.dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(cosine[0].astype(np.float32), np.ones((8,), np.float32))
    np.testing.assert_array_equal(sine[0].astype(np.float32), np.zeros((8,), np.float32))


def _presentation(
    spec: language.MiniMaxH3LanguageConditionerSpec, selected: int
) -> dict[str, object]:
    keyframes = selected // spec.vision_rows_per_keyframe
    rows = max(4, keyframes * (spec.vision_rows_per_keyframe + 2))
    selector = np.zeros((rows, 1), np.int32)
    input_ids = np.zeros((rows,), np.int32)
    for index in range(keyframes):
        start = index * (spec.vision_rows_per_keyframe + 2)
        input_ids[start] = spec.vision_start_token_id
        input_ids[start + 1 : start + 1 + spec.vision_rows_per_keyframe] = spec.image_token_id
        input_ids[start + 1 + spec.vision_rows_per_keyframe] = spec.vision_end_token_id
        selector[start + 1 : start + 1 + spec.vision_rows_per_keyframe] = 1
    bf16 = np.dtype(ml_dtypes.bfloat16)
    return {
        "input_ids": input_ids,
        "mrope_position_ids": np.zeros((3, rows), np.int32),
        "vision_embeddings": np.zeros((rows, spec.hidden_size), bf16),
        "vision_selector": selector,
        "deepstack_embeddings": tuple(
            np.zeros((rows, spec.hidden_size), bf16) for _ in range(spec.deepstack_levels)
        ),
    }


@pytest.mark.parametrize("selected", [0, 2, 4])
def test_presentation_validator_accepts_zero_first_or_both_keyframes(selected: int) -> None:
    spec = _toy_spec()
    assert (
        language.validate_presentation_bindings(spec=spec, **_presentation(spec, selected))
        == selected
    )


def test_presentation_validator_rejects_partial_keyframe_and_bad_positions() -> None:
    spec = _toy_spec()
    partial = _presentation(spec, 0)
    partial["vision_selector"][1] = 1
    partial["input_ids"][1] = spec.image_token_id
    with pytest.raises(ValueError, match="exactly 0, 1008, or 2016"):
        language.validate_presentation_bindings(spec=spec, **partial)

    invalid_selector = _presentation(spec, 0)
    invalid_selector["vision_selector"][0] = 2
    with pytest.raises(ValueError, match="only zero or one"):
        language.validate_presentation_bindings(spec=spec, **invalid_selector)

    bad_position = _presentation(spec, 0)
    bad_position["mrope_position_ids"][0, 0] = spec.max_rows
    with pytest.raises(ValueError, match="must be in"):
        language.validate_presentation_bindings(spec=spec, **bad_position)

    bad_ownership = _presentation(spec, 2)
    bad_ownership["input_ids"][1] = 0
    with pytest.raises(ValueError, match="exactly the image-pad"):
        language.validate_presentation_bindings(spec=spec, **bad_ownership)

    bad_boundary = _presentation(spec, 2)
    bad_boundary["input_ids"][0] = 0
    with pytest.raises(ValueError, match="bounded by vision-start/end"):
        language.validate_presentation_bindings(spec=spec, **bad_boundary)

    non_finite = _presentation(spec, 2)
    non_finite["deepstack_embeddings"][0][1, 0] = ml_dtypes.bfloat16(np.nan)
    with pytest.raises(ValueError, match="must be finite"):
        language.validate_presentation_bindings(spec=spec, **non_finite)


def _ref2va_presentation(
    spec: language.MiniMaxH3LanguageConditionerSpec,
    runs: list[tuple[str, int, int]],
) -> dict[str, object]:
    rows = 1 + sum(length + 2 for _, length, _ in runs)
    input_ids = np.zeros((rows,), np.int32)
    selector = np.zeros((rows, 1), np.int32)
    cursor = 0
    for kind, length, _ in runs:
        input_ids[cursor] = spec.vision_start_token_id
        pad_id = spec.image_token_id if kind == "image" else spec.video_token_id
        input_ids[cursor + 1 : cursor + 1 + length] = pad_id
        selector[cursor + 1 : cursor + 1 + length] = 1
        input_ids[cursor + 1 + length] = spec.vision_end_token_id
        cursor += length + 2
    bf16 = np.dtype(ml_dtypes.bfloat16)
    return {
        "input_ids": input_ids,
        "mrope_position_ids": np.zeros((3, rows), np.int32),
        "vision_embeddings": np.zeros((rows, spec.hidden_size), bf16),
        "vision_selector": selector,
        "deepstack_embeddings": tuple(
            np.zeros((rows, spec.hidden_size), bf16) for _ in range(spec.deepstack_levels)
        ),
        "vision_run_lengths": [length for _, length, _ in runs],
        "vision_run_reference_ids": [reference for _, _, reference in runs],
    }


def test_ref2va_validator_accepts_ordered_variable_image_and_video_runs() -> None:
    spec = _toy_ref2va_spec()
    runs = [
        ("image", 3, 0),
        ("video", 2, 1),
        ("video", 4, 1),
        ("image", 1, 2),
    ]
    bindings = _ref2va_presentation(spec, runs)
    assert language.validate_presentation_bindings(spec=spec, **bindings) == 10


def test_ref2va_validator_requires_exact_runtime_run_metadata() -> None:
    spec = _toy_ref2va_spec()
    bindings = _ref2va_presentation(spec, [("image", 3, 0), ("video", 2, 1)])

    missing = dict(bindings)
    missing["vision_run_lengths"] = None
    with pytest.raises(ValueError, match="runtime-supplied vision_run_lengths"):
        language.validate_presentation_bindings(spec=spec, **missing)

    mismatched = dict(bindings)
    mismatched["vision_run_lengths"] = [2, 3]
    with pytest.raises(ValueError, match="do not match the presentation"):
        language.validate_presentation_bindings(spec=spec, **mismatched)

    reordered = dict(bindings)
    reordered["vision_run_reference_ids"] = [1, 0]
    with pytest.raises(ValueError, match="nondecreasing reference order"):
        language.validate_presentation_bindings(spec=spec, **reordered)

    mixed = _ref2va_presentation(spec, [("image", 2, 0), ("video", 2, 0)])
    with pytest.raises(ValueError, match="cannot mix image and video"):
        language.validate_presentation_bindings(spec=spec, **mixed)


def test_ref2va_validator_enforces_file_and_per_run_caps() -> None:
    spec = _toy_ref2va_spec()

    too_long = _ref2va_presentation(spec, [("image", spec.max_rows, 0)])
    with pytest.raises(ValueError, match=r"presentation rows must be in \[1, 64\]"):
        language.validate_presentation_bindings(spec=spec, **too_long)

    too_many_images = _ref2va_presentation(
        spec, [("image", 1, 0), ("image", 1, 1), ("image", 1, 2)]
    )
    with pytest.raises(ValueError, match="image file cap"):
        language.validate_presentation_bindings(spec=spec, **too_many_images)

    too_many_videos = _ref2va_presentation(
        spec, [("video", 1, 0), ("video", 1, 1), ("video", 1, 2)]
    )
    with pytest.raises(ValueError, match="video file cap"):
        language.validate_presentation_bindings(spec=spec, **too_many_videos)

    oversized_image = _ref2va_presentation(spec, [("image", 7, 0)])
    with pytest.raises(ValueError, match="image run exceeds"):
        language.validate_presentation_bindings(spec=spec, **oversized_image)

    oversized_video = _ref2va_presentation(spec, [("video", 5, 0)])
    with pytest.raises(ValueError, match="video run exceeds"):
        language.validate_presentation_bindings(spec=spec, **oversized_video)

    too_many_video_runs = _ref2va_presentation(spec, [("video", 1, 0)] * (spec.max_video_runs + 1))
    with pytest.raises(ValueError, match="video-duration cap"):
        language.validate_presentation_bindings(spec=spec, **too_many_video_runs)


class _FakeTensor:
    def __init__(
        self,
        name: str,
        dtype: object,
        shape: tuple[int, ...],
        *,
        is_network_input: bool = False,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.is_network_input = is_network_input
        self.dimension_names = {}

    def set_dimension_name(self, axis: int, name: str) -> None:
        if not self.is_network_input:
            raise AssertionError("TensorRT only permits dimension names on network inputs")
        self.dimension_names[axis] = name


class _FakeLayer:
    def __init__(self, output: _FakeTensor):
        self.output = output

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return self.output


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


class _FakeNetwork:
    def __init__(self):
        self.inputs = {}
        self.outputs = []

    def add_input(self, name, dtype, shape):
        tensor = _FakeTensor(name, dtype, shape, is_network_input=True)
        self.inputs[name] = tensor
        return tensor

    def add_gather(self, table, indices, axis):
        del table, indices, axis
        return _FakeLayer(_FakeTensor("token_embeddings", "bf16", (-1, 8)))

    def add_elementwise(self, left, right, operation):
        del operation
        return _FakeLayer(_FakeTensor(f"{left.name}+{right.name}", left.dtype, left.shape))

    def mark_output(self, output):
        self.outputs.append(output)


class _FakeOps:
    @staticmethod
    def weight_constant(network, value):
        del network
        shape = tuple(getattr(value, "shape", (16, 8)))
        return _FakeTensor("embedding_table", "bf16", shape)

    @staticmethod
    def cast(network, tensor, dtype):
        del network
        return _FakeTensor(tensor.name, dtype, tensor.shape)


def _fake_trt() -> SimpleNamespace:
    return SimpleNamespace(int32="int32", bfloat16="bf16", float32="fp32")


def test_dynamic_input_profiles_are_row_aligned_and_unpadded() -> None:
    spec = _toy_spec()
    trt = _fake_trt()
    network = _FakeNetwork()
    inputs = language._declare_inputs(network, spec, trt)
    assert inputs["input_ids"].shape == (-1,)
    assert inputs["mrope_position_ids"].shape == (3, -1)
    assert inputs["vision_embeddings"].shape == (-1, 8)
    assert all(
        tensor.dimension_names[row_axis] == "presentation_rows"
        for name, tensor in inputs.items()
        for row_axis in (1 if name == "mrope_position_ids" else 0,)
    )

    builder = _FakeBuilder()
    config = _FakeConfig()
    language._add_optimization_profile(builder, config, spec)
    assert len(config.profiles) == 1
    assert builder.profile.shapes["input_ids"] == ((1,), (4,), (16,))
    assert builder.profile.shapes["mrope_position_ids"] == (
        (3, 1),
        (3, 4),
        (3, 16),
    )
    assert builder.profile.shapes["vision_embeddings"] == (
        (1, 8),
        (4, 8),
        (16, 8),
    )
    assert "attention_mask" not in builder.profile.shapes

    ref_spec = _toy_ref2va_spec()
    ref_builder = _FakeBuilder()
    ref_config = _FakeConfig()
    language._add_optimization_profile(ref_builder, ref_config, ref_spec)
    assert ref_builder.profile.shapes["input_ids"] == ((1,), (8,), (64,))
    assert ref_builder.profile.shapes["mrope_position_ids"] == (
        (3, 1),
        (3, 8),
        (3, 64),
    )
    assert "attention_mask" not in ref_builder.profile.shapes


def test_dynamic_mrope_preserves_bf16_publication_boundaries(monkeypatch) -> None:
    spec = _toy_spec()
    trt = SimpleNamespace(bfloat16="bf16", float32="fp32")
    tensor = _FakeTensor("q", "bf16", (-1, spec.num_heads * spec.head_dim))
    cosine = _FakeTensor("cos", "bf16", (spec.max_rows, spec.head_dim // 2))
    sine = _FakeTensor("sin", "bf16", (spec.max_rows, spec.head_dim // 2))
    casts = []
    observed = {}

    class Ops:
        @staticmethod
        def cast(network, value, dtype):
            del network
            casts.append(dtype)
            return _FakeTensor(value.name, dtype, value.shape)

    class Network:
        @staticmethod
        def add_rotary_embedding(value, cos, sin, interleaved, rotary_dim):
            observed.update(
                value_dtype=value.dtype,
                cos_dtype=cos.dtype,
                sin_dtype=sin.dtype,
                interleaved=interleaved,
                rotary_dim=rotary_dim,
            )
            return _FakeLayer(_FakeTensor("rotated", value.dtype, value.shape))

    monkeypatch.setattr(language, "_rows_to_heads", lambda *args: args[1])
    monkeypatch.setattr(language, "_heads_to_rows", lambda *args: args[1])
    monkeypatch.setattr(language, "_select_mrope_cache", lambda *args: args[1])
    output = language._apply_mrope(
        Network(), tensor, (cosine, sine), object(), spec.num_heads, spec, trt, Ops
    )
    assert casts == ["bf16", "bf16"]
    assert observed == {
        "value_dtype": "bf16",
        "cos_dtype": "bf16",
        "sin_dtype": "bf16",
        "interleaved": False,
        "rotary_dim": spec.head_dim,
    }
    assert output.dtype == "bf16"


def test_mocked_graph_replaces_main_and_injects_after_layers_zero_one_two(
    monkeypatch,
) -> None:
    spec = _toy_spec()
    trt = SimpleNamespace(
        int32="int32",
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(SUM="sum"),
    )
    network = _FakeNetwork()
    inputs = language._declare_inputs(network, spec, trt)
    layer_inputs = []
    gate_inputs = []

    monkeypatch.setattr(language, "_selector_condition", lambda *args: object())
    monkeypatch.setattr(
        language,
        "_hard_select",
        lambda network, condition, vision, token: _FakeTensor(
            "vision-replaced", "bf16", (-1, spec.hidden_size)
        ),
    )
    monkeypatch.setattr(
        language,
        "_make_rope_tables",
        lambda current: (
            np.zeros((current.max_rows, current.head_dim // 2), np.float32),
            np.zeros((current.max_rows, current.head_dim // 2), np.float32),
        ),
    )

    def layer(network, hidden, weights, index, rope, positions, current, trt, op):
        del network, weights, rope, positions, current, trt, op
        layer_inputs.append((index, hidden.name))
        return _FakeTensor(f"layer-{index}", "bf16", (-1, spec.hidden_size))

    monkeypatch.setattr(language, "_language_layer", layer)

    def gate(network, condition, value, trt, op):
        del network, condition, trt, op
        gate_inputs.append(value.name)
        return _FakeTensor(f"gated-{value.name}", "bf16", (-1, spec.hidden_size))

    monkeypatch.setattr(language, "_hard_gate", gate)
    weights = {"model.language_model.embed_tokens.weight": _ShapeOnly((16, 8))}
    output = language._assemble_language_conditioner_graph(
        network, weights, spec, inputs, trt, _FakeOps
    )
    assert layer_inputs == [
        (0, "vision-replaced"),
        (1, "layer-0+gated-deepstack_embeddings_0"),
        (2, "layer-1+gated-deepstack_embeddings_1"),
        (3, "layer-2+gated-deepstack_embeddings_2"),
    ]
    assert gate_inputs == [
        "deepstack_embeddings_0",
        "deepstack_embeddings_1",
        "deepstack_embeddings_2",
    ]
    assert output.name == "encoder_hidden_states"
    assert output.dtype == "fp32"
    assert network.outputs == [output]


def _tiny_weights(spec: language.MiniMaxH3LanguageConditionerSpec) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260824)
    return {
        name: (
            np.ones(shape, np.float32)
            if name.endswith("norm.weight")
            else rng.normal(0.0, 0.02, shape).astype(np.float32)
        )
        for name, shape in language.expected_weight_shapes(spec).items()
    }


@pytest.mark.gpu
def test_tiny_dynamic_language_graph_serializes_when_tensorrt_is_available() -> None:
    trt = pytest.importorskip("tensorrt")
    try:
        probe_builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    except Exception as error:
        pytest.skip(f"TensorRT builder initialization is unavailable: {error}")
    if probe_builder is None:
        pytest.skip("TensorRT builder initialization returned null")
    del probe_builder
    spec = _toy_spec(output_layers=3)
    plan = language._build_language_conditioner_engine(
        _tiny_weights(spec),
        spec,
        verbose=False,
        consume_weights=False,
        workspace_bytes=1 << 30,
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_shape("input_ids") == (-1,)
    assert engine.get_tensor_shape("mrope_position_ids") == (3, -1)
    assert engine.get_tensor_shape("encoder_hidden_states") == (-1, 8)
    assert tuple(engine.get_tensor_profile_shape("input_ids", 0)) == (
        (1,),
        (4,),
        (16,),
    )
