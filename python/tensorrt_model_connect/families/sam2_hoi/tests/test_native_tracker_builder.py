# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi import native_tracker_builder


def test_native_tracker_builder_is_dependency_light_and_has_no_onnx_path():
    path = Path(native_tracker_builder.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            top_level_imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            top_level_imports.append(statement.module)

    assert "onnx" not in source.lower()
    assert "families.sam3" not in source
    assert "source_export" not in source
    assert "state_dict" not in source
    assert not any(name == "torch" or name.startswith("torch.") for name in top_level_imports)
    assert {
        "create_network",
        "add_attention",
        "add_convolution_nd",
        "add_matrix_multiply",
        "add_reduce",
        "build_serialized_network",
    } <= calls
    assert "Sam2HoiLayerNorm256" in source
    assert "Sam2HoiSigmoid" in source


def test_prompt_encoder_preserves_reviewed_bf16_and_fp32_boundaries():
    dense_source = inspect.getsource(native_tracker_builder._dense_random_position_encoding)
    dense_values_source = inspect.getsource(
        native_tracker_builder._dense_random_position_encoding_values
    )
    point_source = inspect.getsource(native_tracker_builder._point_prompt_embeddings)
    k2_source = inspect.getsource(native_tracker_builder._k2_projection)
    empty_source = inspect.getsource(native_tracker_builder._empty_prompt_embeddings)
    no_mask_source = inspect.getsource(native_tracker_builder._no_mask_dense_embeddings)
    decoder_source = inspect.getsource(native_tracker_builder._add_mask_decoder)

    assert "_constant_for_dtype" in dense_source
    assert "add_matrix_multiply" not in dense_source
    assert "add_unary" not in dense_source
    assert "_round_float32_to_bfloat16" in dense_values_source
    assert "np.float32(2.0 * np.pi)" in dense_values_source
    assert "np.sin" in dense_values_source
    assert "np.cos" in dense_values_source
    assert "add_matrix_multiply" not in point_source
    assert "_k2_projection" in point_source
    assert "np.full((1, 1, 1), 2.0 * np.pi, dtype=np.float32)" in point_source
    assert "_cast(network, phases, trt.float32)" in point_source
    assert "_cast(network, tau, learned_dtype)" not in point_source
    assert "add_matrix_multiply" not in k2_source
    assert k2_source.count("add_slice") == 4
    assert "ElementWiseOperation.PROD" in k2_source
    assert "ElementWiseOperation.SUM" in k2_source
    assert "base_fp32 = _cast(network, base, trt.float32)" in point_source
    assert "not_a_point = _constant(" in point_source
    assert "learned = _constant(" in point_source
    assert "padding = _constant(" in point_source
    assert "return _constant(network" in empty_source
    assert "return _constant(network" in no_mask_source
    assert "output_tokens = _constant(" in decoder_source


def test_dense_random_position_encoding_bf16_constant_is_deterministic():
    key = "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    weights = {key: np.linspace(-1.25, 1.25, 2 * 128, dtype=np.float32).reshape(2, 128)}

    values = native_tracker_builder._dense_random_position_encoding_values(
        weights,
        bf16=True,
    )
    bits = (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)

    assert values.shape == (1, 256, 64, 64)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(
        values,
        native_tracker_builder._round_float32_to_bfloat16(values),
    )
    assert hashlib.sha256(bits.tobytes()).hexdigest() == (
        "8a61d5cfb44a12fef51b202ec57a8b219480a69ae4a5abb2b7b1bf6cf58cc2bb"
    )


def test_software_bfloat16_rounding_uses_round_to_nearest_even():
    values = np.asarray(
        [0x3F807FFF, 0x3F808000, 0x3F808001, 0x3F818000],
        dtype=np.uint32,
    ).view(np.float32)

    rounded = native_tracker_builder._round_float32_to_bfloat16(values)

    np.testing.assert_array_equal(
        rounded.view(np.uint32),
        np.asarray([0x3F800000, 0x3F800000, 0x3F810000, 0x3F820000], dtype=np.uint32),
    )


def test_reviewed_hand_stability_fixture_selects_best_multimask_token():
    # Exact L4 source prompt-frame counts: token 0 is below the reviewed 0.98
    # stability boundary, while token 2 has the highest multimask IoU estimate.
    intersection = 138
    union = 142
    all_ious = np.asarray([0.71484375, 0.6796875, 0.734375, 0.703125])
    prompt_source = inspect.getsource(native_tracker_builder._prompt_head)

    stability = intersection / union
    best_multimask = 1 + int(np.argmax(all_ious[1:]))
    selected = 0 if stability >= 0.98 else best_multimask

    assert "np.full((1, 1), 0.98, dtype=np.float32)" in prompt_source
    assert "network.add_select(stable_mask, single_mask, best.mask)" in prompt_source
    assert stability == pytest.approx(0.971830985915493)
    assert best_multimask == 2
    assert selected == 2


def test_layer_norm2d_uses_source_ordered_native_reductions():
    source = inspect.getsource(native_tracker_builder._layer_norm_channels)

    assert source.count("ReduceOperation.AVG") == 2
    assert "centered_fp32 = _cast(network, centered, trt.float32)" in source
    assert "UnaryOperation.SQRT" in source
    assert "ElementWiseOperation.DIV" in source
    assert "add_normalization" not in source
    assert "Sam2HoiLayerNorm256" not in source


def test_nn_layer_norm_and_cxblock_use_qualified_dtype_paths():
    layer_norm_source = inspect.getsource(native_tracker_builder._layer_norm_last)
    fuser_source = inspect.getsource(native_tracker_builder._memory_fuser_block)

    assert "Sam2HoiLayerNorm256" in layer_norm_source
    assert "add_normalization" not in layer_norm_source
    assert "residual = tensor" in fuser_source
    assert "gamma = _constant(" in fuser_source
    assert "_cast(network, tensor, trt.float32)" in fuser_source
    assert "return _fp32_sum(network, residual, tensor)" in fuser_source


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_tracker_bindings_preserve_runtime_contract(precision):
    prompt, recurrent, memory = native_tracker_builder.tracker_binding_specs(precision)
    work = "bfloat16" if precision == "bf16" else "float32"

    assert prompt.section == "sam2_hoi_prompt_tracker_engine_plan"
    assert [(item.name, item.dtype, item.shape) for item in prompt.inputs] == [
        ("tracker_feature_0", work, (1, 32, 256, 256)),
        ("tracker_feature_1", work, (1, 64, 128, 128)),
        ("tracker_feature_2", "float32", (1, 256, 64, 64)),
        ("point_coords", "float32", (2, 3, 2)),
        ("point_labels", "int32", (2, 3)),
    ]
    assert [(item.name, item.shape) for item in prompt.outputs] == [
        ("pred_masks", (2, 1, 256, 256)),
        ("object_pointer", (2, 256)),
        ("object_score_logits", (2, 1)),
        ("selected_iou", (2, 1)),
    ]

    recurrent_inputs = {item.name: item for item in recurrent.inputs}
    assert recurrent_inputs["memory_features"].shape == (2, -1, 64, 64, 64)
    assert recurrent_inputs["memory_features"].profile == (
        (2, 1, 64, 64, 64),
        (2, 3, 64, 64, 64),
        (2, 7, 64, 64, 64),
    )
    assert recurrent_inputs["object_pointers"].profile == (
        (2, 1, 256),
        (2, 2, 256),
        (2, 16, 256),
    )
    assert recurrent.outputs[-1].shape == (2, 3)

    output_dtype = "bfloat16" if precision == "bf16" else "float32"
    assert [(item.name, item.dtype, item.shape) for item in memory.outputs] == [
        ("new_memory_features", output_dtype, (2, 64, 64, 64)),
        ("new_memory_position", output_dtype, (2, 64, 64, 64)),
    ]


def test_native_tracker_rope_and_memory_position_constants_are_exactly_sized():
    cosine, sine = native_tracker_builder._axial_rope_arrays()
    assert cosine.shape == sine.shape == (64 * 64, 256)
    np.testing.assert_allclose(cosine * cosine + sine * sine, 1.0, rtol=2e-6, atol=2e-6)

    position = native_tracker_builder._memory_position_encoding()
    assert position.shape == (2, 64, 64, 64)
    assert position.dtype == np.float32
    assert np.isfinite(position).all()
    np.testing.assert_array_equal(position[0], position[1])


def _fake_native_tracker_weights():
    weights = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in native_tracker_builder._ARCHITECTURE_WEIGHT_SHAPES
    }
    for prefix, indices in native_tracker_builder._ARCHITECTURE_LAYER_INDICES:
        for index in indices:
            weights.setdefault(f"{prefix}{index}.sentinel", np.zeros((), dtype=np.float32))
    return weights


def test_native_tracker_weight_validation_is_fail_closed():
    weights = _fake_native_tracker_weights()
    native_tracker_builder._validate_native_tracker_weights(weights)

    weights["maskmem_tpos_enc"] = np.zeros((8, 1, 1, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="maskmem_tpos_enc.*expected.*7, 1, 1, 64"):
        native_tracker_builder._validate_native_tracker_weights(weights)

    weights = _fake_native_tracker_weights()
    weights["memory_attention.layers.4.sentinel"] = np.zeros((), dtype=np.float32)
    with pytest.raises(ValueError, match="memory_attention.layers.*expected.*0, 1, 2, 3"):
        native_tracker_builder._validate_native_tracker_weights(weights)


def test_native_tracker_precision_rejects_unqualified_modes():
    with pytest.raises(ValueError, match="supports.*bf16.*fp32"):
        native_tracker_builder.tracker_binding_specs("fp16")
