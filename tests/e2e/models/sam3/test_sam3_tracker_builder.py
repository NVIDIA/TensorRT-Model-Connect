# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the model-owned SAM3.0 tracker plan builder."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.sam3 import tracker_builder, tracker_weights


def test_sam3_tracker_step_profiles_cover_native_memory_bounds() -> None:
    assert tracker_builder._step_profile_shapes("memory_features") == (
        (1, 1, 5184, 64),
        (1, 3, 5184, 64),
        (1, 10, 5184, 64),
    )
    assert tracker_builder._step_profile_shapes("memory_temporal_offsets") == (
        (1, 1),
        (1, 3),
        (1, 10),
    )
    assert tracker_builder._step_profile_shapes("object_pointers") == (
        (1, 1, 256),
        (1, 4, 256),
        (1, 19, 256),
    )
    assert tracker_builder._step_profile_shapes("object_pointer_temporal_offsets") == (
        (1, 1),
        (1, 4),
        (1, 19),
    )
    assert tracker_builder._step_profile_shapes("tracker_feature_2") is None


def test_sam3_tracker_batch2_step_profiles_use_measured_opt_shapes() -> None:
    assert tracker_builder._step_batch2_profile_shapes("memory_features") == (
        (2, 1, 5184, 64),
        (2, 3, 5184, 64),
        (2, 10, 5184, 64),
    )
    assert tracker_builder._step_batch2_profile_shapes("memory_temporal_offsets") == (
        (2, 1),
        (2, 3),
        (2, 10),
    )
    assert tracker_builder._step_batch2_profile_shapes("object_pointers") == (
        (2, 1, 256),
        (2, 2, 256),
        (2, 19, 256),
    )
    assert tracker_builder._step_batch2_profile_shapes("object_pointer_temporal_offsets") == (
        (2, 1),
        (2, 2),
        (2, 19),
    )
    assert tracker_builder._step_batch2_profile_shapes("tracker_feature_2") is None


def _official_tracker_config() -> dict[str, object]:
    return {
        "image_size": 1008,
        "vision_config": {
            "backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]],
            "fpn_hidden_size": 256,
            "num_feature_levels": 3,
        },
        "prompt_encoder_config": {
            "hidden_size": 256,
            "image_size": 1008,
            "patch_size": 14,
            "mask_input_channels": 16,
            "num_point_embeddings": 4,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu",
            "scale": 1,
        },
        "mask_decoder_config": {
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "attention_downsample_rate": 2,
            "mlp_dim": 2048,
            "num_multimask_outputs": 3,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 256,
        },
        "num_maskmem": 7,
        "max_cond_frame_num": 4,
        "max_object_pointers_in_encoder": 16,
        "memory_attention_hidden_size": 256,
        "memory_attention_num_attention_heads": 1,
        "memory_attention_num_layers": 4,
        "memory_attention_feed_forward_hidden_size": 2048,
        "memory_attention_feed_forward_hidden_act": "relu",
        "memory_attention_downsample_rate": 1,
        "memory_attention_rope_feat_sizes": [72, 72],
        "memory_attention_rope_theta": 10000,
        "memory_encoder_hidden_size": 256,
        "memory_encoder_output_channels": 64,
        "mask_downsampler_embed_dim": 256,
        "mask_downsampler_hidden_act": "gelu",
        "mask_downsampler_kernel_size": 3,
        "mask_downsampler_padding": 1,
        "mask_downsampler_stride": 2,
        "mask_downsampler_total_stride": 16,
        "memory_fuser_embed_dim": 256,
        "memory_fuser_hidden_act": "gelu",
        "memory_fuser_intermediate_dim": 1024,
        "memory_fuser_kernel_size": 7,
        "memory_fuser_padding": 3,
        "memory_fuser_num_layers": 2,
        "memory_fuser_layer_scale_init_value": 1e-6,
        "sigmoid_scale_for_mem_enc": 20.0,
        "sigmoid_bias_for_mem_enc": -10.0,
        "enable_occlusion_spatial_embedding": True,
        "enable_temporal_pos_encoding_for_object_pointers": True,
        "multimask_output_for_tracking": True,
        "multimask_output_in_sam": True,
        "multimask_min_pt_num": 0,
        "multimask_max_pt_num": 1,
    }


def _set_config_path(config: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = config
    for name in path[:-1]:
        nested = current[name]
        assert isinstance(nested, dict)
        current = nested
    current[path[-1]] = value


def test_sam3_tracker_validation_accepts_official_sam3_architecture() -> None:
    tracker_builder._validate_tracker_model(SimpleNamespace(config=_official_tracker_config()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_size", 1024, "1008px tracker only"),
        ("num_maskmem", 8, "memory profile must match"),
        ("max_cond_frame_num", 5, "conditioning-memory profile must match"),
        ("max_object_pointers_in_encoder", 32, "pointer profile must match"),
    ],
)
def test_sam3_tracker_validation_rejects_unreviewed_variants(
    field: str, value: int, message: str
) -> None:
    config = _official_tracker_config()
    config[field] = value
    with pytest.raises(RuntimeError, match=message):
        tracker_builder._validate_tracker_model(SimpleNamespace(config=config))


@pytest.mark.parametrize(
    ("path", "unreviewed_value"),
    [
        (path, False if expected is True else "unreviewed")
        for path, expected in tracker_builder._TRACKER_ARCHITECTURE_CONTRACT
    ],
)
def test_sam3_tracker_validation_rejects_every_unreviewed_graph_config(
    path: tuple[str, ...], unreviewed_value: object
) -> None:
    config = _official_tracker_config()
    _set_config_path(config, path, unreviewed_value)

    with pytest.raises(RuntimeError, match="supports only the official architecture"):
        tracker_builder._validate_tracker_model(SimpleNamespace(config=config))


def test_sam3_tracker_model_config_rejects_unreviewed_low_resolution(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "low_res_mask_size": 288,
                "tracker_config": _official_tracker_config(),
            }
        ),
        encoding="utf-8",
    )
    tracker_builder._read_model_config(str(tmp_path))

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["low_res_mask_size"] = 256
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected low_res_mask_size=288"):
        tracker_builder._read_model_config(str(tmp_path))


def test_sam3_tracker_bundle_section_names_are_stable() -> None:
    assert tracker_builder.TRACKER_INIT_SECTION == "sam3_tracker_init_engine_plan"
    assert tracker_builder.TRACKER_STEP_SECTION == "sam3_tracker_step_engine_plan"
    assert tracker_builder.TRACKER_STEP_BATCH2_SECTION == "sam3_tracker_step_batch2_engine_plan"
    assert tracker_builder.TRACKER_MEMORY_SECTION == "sam3_tracker_memory_engine_plan"
    assert tracker_builder.TRACKER_MEMORY_BATCH2_SECTION == "sam3_tracker_memory_batch2_engine_plan"
    assert tracker_builder.TRACKER_HARD_MEMORY_SECTION == "sam3_tracker_hard_memory_engine_plan"
    assert (
        tracker_builder.TRACKER_HARD_MEMORY_BATCH2_SECTION
        == "sam3_tracker_hard_memory_batch2_engine_plan"
    )


def test_sam3_tracker_init_uses_meta_mask_input_geometry() -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    assert tracker_decoder_builder._LOW_RESOLUTION == 288
    assert tracker_decoder_builder._MASK_INPUT_SIZE == 1152

    source = inspect.getsource(tracker_decoder_builder.add_tracker_init_head)
    assert "resized_detector_logits" in source
    assert "ElementWiseOperation.GREATER" in source
    assert "target_height=_MASK_INPUT_SIZE" in source
    assert "input_size=_MASK_INPUT_SIZE" in source
    assert "mask_max = network.add_reduce(\n        mask_input," in source


def test_sam3_tracker_init_pointer_uses_meta_single_mask_token() -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    decoder_source = inspect.getsource(tracker_decoder_builder.add_mask_decoder)
    assert "single_mask_token=single_token.get_output(0)" in decoder_source

    init_source = inspect.getsource(tracker_decoder_builder.add_tracker_init_head)
    assert "decoder_outputs.single_mask_token" in init_source
    assert "selected.mask_token" not in init_source


def test_sam3_tracker_decoder_precision_policy_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    fake_trt = SimpleNamespace(float32="fp32")
    monkeypatch.setattr(tracker_decoder_builder, "_trt", lambda: fake_trt)

    policy = tracker_decoder_builder.DecoderPrecisionPolicy
    assert tracker_decoder_builder._learned_dtype(policy.INIT_FP32) == "fp32"
    assert tracker_decoder_builder._learned_dtype(policy.RECURRENT_TRT_FP32) == "fp32"
    with pytest.raises(TypeError, match="must be a DecoderPrecisionPolicy"):
        tracker_decoder_builder._learned_dtype("bf16")

    required_policy_helpers = (
        tracker_decoder_builder._linear,
        tracker_decoder_builder._attention,
        tracker_decoder_builder._feed_forward,
        tracker_decoder_builder._decoder_mlp,
        tracker_decoder_builder._two_way_transformer,
        tracker_decoder_builder._add_deconvolution,
        tracker_decoder_builder.add_mask_decoder,
        tracker_decoder_builder.add_object_pointer_projection,
    )
    for helper in required_policy_helpers:
        parameter = inspect.signature(helper).parameters["precision_policy"]
        assert parameter.default is inspect.Parameter.empty

    deconvolution_source = inspect.getsource(tracker_decoder_builder._add_deconvolution)
    assert "network.add_deconvolution_nd(" in deconvolution_source
    assert "kernel=trt.Weights(weight)" in deconvolution_source
    assert "bias=trt.Weights(bias)" in deconvolution_source
    assert "network.add_matrix_multiply(" not in deconvolution_source
    assert "_cast(network, _cast(network, inp" not in deconvolution_source


def test_sam3_tracker_heads_select_phase_specific_decoder_precision() -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    init_source = inspect.getsource(tracker_decoder_builder.add_tracker_init_head)
    assert "precision_policy = DecoderPrecisionPolicy.INIT_FP32" in init_source
    assert init_source.count("precision_policy=precision_policy") == 2

    step_source = inspect.getsource(tracker_decoder_builder.add_tracker_step_head)
    assert "precision_policy = DecoderPrecisionPolicy.RECURRENT_TRT_FP32" in step_source
    assert step_source.count("precision_policy=precision_policy") == 2

    decoder_source = inspect.getsource(tracker_decoder_builder.add_mask_decoder)
    assert "learned_dtype = _learned_dtype(precision_policy)" in decoder_source
    assert "trt.float32" in decoder_source
    assert "_precision_constant(" in decoder_source
    assert "precision_policy=precision_policy" in decoder_source
    assert "fp16_attention" not in decoder_source


def test_sam3_tracker_step_plan_preserves_meta_selected_iou_output() -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    decoder_source = inspect.getsource(tracker_decoder_builder.add_mask_decoder)
    assert "iou_prediction_head" in decoder_source
    iou_head = decoder_source.index('"mask_decoder.iou_prediction_head"')
    sigmoid = decoder_source.index("trt.ActivationType.SIGMOID", iou_head)
    reshape = decoder_source.index("iou_scores_shape =", sigmoid)
    assert iou_head < sigmoid < reshape

    selection_source = inspect.getsource(tracker_decoder_builder.add_multimask_selection)
    assert "decoder_outputs.iou_scores" in selection_source
    bf16_round = selection_source.index("_cast(network, decoder_outputs.iou_scores, trt.bfloat16)")
    fp32_publish = selection_source.index("trt.float32", bf16_round)
    first_candidate = selection_source.index("best_score = network.add_slice", fp32_publish)
    assert bf16_round < fp32_publish < first_candidate
    assert "iou_score=best_score" in selection_source

    head_source = inspect.getsource(tracker_decoder_builder.add_tracker_step_head)
    assert "selected_iou=selected.iou_score" in head_source

    build_source = inspect.getsource(tracker_builder._build_step)
    assert "selected_iou = head.selected_iou" in build_source
    assert '_mark(network, selected_iou, "selected_iou")' in build_source


def test_sam3_tracker_multimask_selection_preserves_first_equal_iou_candidate() -> None:
    from tensorrt_model_connect.families.sam3 import tracker_decoder_builder

    source = inspect.getsource(tracker_decoder_builder.add_multimask_selection)
    first_candidate = source.index("best_score = network.add_slice")
    first_index = source.index("np.zeros((1, 1), dtype=np.int32)", first_candidate)
    scan = source.index("for candidate_index in range(1, _NUM_MULTIMASKS):", first_index)
    strict_compare = source.index("trt.ElementWiseOperation.GREATER", scan)
    score_select = source.index("best_score = network.add_select(", strict_compare)
    index_select = source.index("best_indices = network.add_select(", score_select)

    assert first_candidate < first_index < scan < strict_compare < score_select < index_select
    assert "TopKOperation" not in source
    assert "GREATER_OR_EQUAL" not in source


def test_sam3_tracker_init_defers_recurrent_memory_until_global_frame_zero_policy() -> None:
    init_source = inspect.getsource(tracker_builder._build_init)
    assert "add_tracker_memory_encoder" not in init_source
    assert '"memory_features"' not in init_source
    assert '"memory_position"' not in init_source
    assert '"pred_masks"' not in init_source


def test_sam3_tracker_hard_memory_plan_has_fixed_b1_b2_contract() -> None:
    memory_plan_source = inspect.getsource(tracker_builder._build_hard_memory)
    assert '"tracker_feature_2"' in memory_plan_source
    assert '"owned_tracker_mask"' in memory_plan_source
    assert "(batch_size, 1, 1008, 1008)" in memory_plan_source
    assert '"object_score_logits"' in memory_plan_source
    assert "hard_mask=True" in memory_plan_source
    assert '_mark(network, outputs.memory, "new_memory_features")' in memory_plan_source
    assert '_mark(network, outputs.position, "new_memory_position")' in memory_plan_source


def test_sam3_tracker_hard_memory_matches_meta_geometry_order() -> None:
    memory_path = Path(tracker_builder.__file__).with_name("tracker_memory_builder.py")
    module_source = memory_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(memory_path))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert constants["_LOW_RES_MASK_SIZE"] == 288
    assert constants["_TRACKER_IMAGE_SIZE"] == 1008
    assert constants["_MEMORY_MASK_SIZE"] == 1152

    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    source = functions["_prepare_memory_mask"]
    assert source is not None
    resize_to_tracker = source.index("_TRACKER_IMAGE_SIZE")
    non_overlap = source.index("_apply_hard_mask_non_overlap")
    threshold = source.index("ElementWiseOperation.GREATER")
    scale = source.index("_SIGMOID_SCALE")
    resize_to_memory = source.rindex("_MEMORY_MASK_SIZE")
    assert resize_to_tracker < non_overlap < threshold < scale < resize_to_memory
    assert "PyTorch's antialias support is 1" in source

    non_overlap_source = functions["_apply_hard_mask_non_overlap"]
    assert non_overlap_source is not None
    assert "TopKOperation.MAX" in non_overlap_source
    assert "1 << 0" in non_overlap_source
    assert "ElementWiseOperation.EQUAL" in non_overlap_source
    assert "network.add_select" in non_overlap_source


def test_sam3_tracker_soft_memory_matches_meta_geometry_order() -> None:
    memory_path = Path(tracker_builder.__file__).with_name("tracker_memory_builder.py")
    module_source = memory_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(memory_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    source = functions["_prepare_memory_mask"]
    assert source is not None
    soft_branch = source[source.index("# The dense-video update") :]
    resize_to_memory = soft_branch.index("_MEMORY_MASK_SIZE")
    suppress_after_resize = soft_branch.index("_apply_soft_area_shrinkage")
    sigmoid = soft_branch.index("ActivationType.SIGMOID")
    scale = soft_branch.index("_SIGMOID_SCALE")
    bias = soft_branch.index("_SIGMOID_BIAS")
    return_without_resize = soft_branch.index("if not hard_mask:\n        return prepared")
    assert resize_to_memory < suppress_after_resize < sigmoid < scale < bias < return_without_resize
    assert "_TRACKER_IMAGE_SIZE" not in soft_branch[:return_without_resize]

    suppression_source = functions["_apply_soft_area_shrinkage"]
    assert suppression_source is not None
    assert "(batch_size, 1, 1, 1)" in suppression_source
    assert "ElementWiseOperation.GREATER" in suppression_source
    assert "ElementWiseOperation.MIN" in suppression_source
    assert "network.add_select(reject, clamped, high_res_mask_logits)" in suppression_source

    memory_plan_source = inspect.getsource(tracker_builder._build_memory)
    assert '"suppress_area_shrinkage"' in memory_plan_source
    assert "trt.int32" in memory_plan_source
    assert "(batch_size, 1)" in memory_plan_source
    assert "hard_mask=False" in memory_plan_source

    public_source = functions["add_tracker_memory_encoder"]
    assert public_source is not None
    assert "suppress_area_shrinkage: trt.ITensor | None = None" in public_source
    assert "suppress_area_shrinkage" in public_source


def test_sam3_tracker_memory_default_uses_direct_tensorrt_network_api() -> None:
    memory_path = Path(tracker_builder.__file__).with_name("tracker_memory_builder.py")
    module_source = memory_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(memory_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    public_source = functions["add_tracker_memory_encoder"]
    assert public_source is not None
    assert "compute_precision" not in public_source
    assert "dtype:" not in public_source
    assert "_add_bf16_conv2d" in public_source
    assert "_add_mask_downsampler" in public_source
    assert "_add_convnext_fuser_block" in public_source
    assert "_add_occlusion_embedding" in public_source

    build_source = inspect.getsource(tracker_builder.build_sam3_tracker_engines)
    assert "compute_precision" not in build_source
    assert "_build_memory(" in build_source
    assert "_build_hard_memory(" in build_source
    assert "build_sam3_tracker_memory_ffi_plans" not in build_source
    assert "memory_plan_spec" not in build_source
    assert "aoti" not in build_source.lower()
    assert "tvm" not in build_source.lower()
    assert hasattr(tracker_builder, "_build_memory")
    assert hasattr(tracker_builder, "_build_hard_memory")


def test_sam3_tracker_memory_native_bf16_ops_keep_fp32_abi() -> None:
    memory_path = Path(tracker_builder.__file__).with_name("tracker_memory_builder.py")
    module_source = memory_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(memory_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    conv_source = functions["_add_bf16_conv2d"]
    assert conv_source is not None
    assert "_cast(network, inp, trt.bfloat16)" in conv_source
    assert "kernel=trt.Weights()" in conv_source
    assert "bias=trt.Weights()" in conv_source
    assert "convolution.set_input(1, _bf16_constant" in conv_source
    assert "convolution.set_input(\n            2," in conv_source

    linear_source = functions["_add_bf16_linear"]
    assert linear_source is not None
    assert "_cast(network, inp, trt.bfloat16)" in linear_source
    assert "_bf16_constant" in linear_source
    assert "network.add_matrix_multiply" in linear_source
    assert "bias_tensor" not in linear_source
    assert "ElementWiseOperation.SUM" not in linear_source

    output_source = functions["_format_outputs"]
    assert output_source is not None
    assert "memory_output = _cast(network, memory_output, trt.float32)" in output_source
    assert "position_output = add_constant" in output_source
    position_constant = output_source.index("position_output = add_constant")
    position_bf16 = output_source.index(
        "position_output = _cast(network, position_output, trt.bfloat16)"
    )
    position_fp32_carrier = output_source.index(
        "position_output = _cast(network, position_output, trt.float32)"
    )
    assert position_constant < position_bf16 < position_fp32_carrier


def test_sam3_tracker_memory_matches_inductor_fusion_and_occlusion_rounding() -> None:
    memory_path = Path(tracker_builder.__file__).with_name("tracker_memory_builder.py")
    module_source = memory_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(memory_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    layer_norm_source = functions["_add_inductor_layer_norm_2d"]
    assert layer_norm_source is not None
    fp32_input = layer_norm_source.index("_cast(network, inp, trt.float32)")
    fp32_conv_bias = layer_norm_source.index("convolution_bias_tensor")
    fp32_mean = layer_norm_source.index("ReduceOperation.AVG")
    variance_center = layer_norm_source.index("centered_for_variance")
    fp32_square = layer_norm_source.index("ElementWiseOperation.PROD")
    fp32_variance = layer_norm_source.rindex("ReduceOperation.AVG")
    optional_bf16_mean = layer_norm_source.index("_cast(network, mean, trt.bfloat16)")
    numerator_center = layer_norm_source.index("centered = network.add_elementwise")
    fp32_sqrt = layer_norm_source.index("UnaryOperation.SQRT")
    fp32_divide = layer_norm_source.index("ElementWiseOperation.DIV")
    assert (
        fp32_input
        < fp32_conv_bias
        < fp32_mean
        < variance_center
        < fp32_square
        < fp32_variance
        < optional_bf16_mean
        < numerator_center
        < fp32_sqrt
        < fp32_divide
    )

    downsampler_source = functions["_add_mask_downsampler"]
    assert downsampler_source is not None
    bias_free_conv = downsampler_source.index("None,\n            out_channels")
    fused_conv_bias = downsampler_source.index('_weight(weights, f"{prefix}.conv.bias")')
    gelu = downsampler_source.index("add_gelu_erf")
    bf16_publish = downsampler_source.index("_cast(network, hidden, trt.bfloat16)")
    assert bias_free_conv < fused_conv_bias < gelu < bf16_publish
    assert "round_mean_bf16=layer_index == 0" in downsampler_source

    fuser_source = functions["_add_convnext_fuser_block"]
    assert fuser_source is not None
    assert 'f"{prefix}.depthwise_conv.weight"),\n        None,' in fuser_source
    assert "_add_fp32_bias_gelu_to_bf16" in fuser_source
    assert "_add_fp32_bias_scale_residual_to_bf16" in fuser_source

    gelu_epilogue_source = functions["_add_fp32_bias_gelu_to_bf16"]
    assert gelu_epilogue_source is not None
    bias_add = gelu_epilogue_source.index("bias_tensor")
    fp32_gelu = gelu_epilogue_source.index("add_gelu_erf")
    bf16_gelu_publish = gelu_epilogue_source.index("trt.bfloat16")
    assert bias_add < fp32_gelu < bf16_gelu_publish

    residual_epilogue_source = functions["_add_fp32_bias_scale_residual_to_bf16"]
    assert residual_epilogue_source is not None
    bias_add = residual_epilogue_source.index("bias_tensor")
    scale_mul = residual_epilogue_source.index("scale_tensor")
    residual_add = residual_epilogue_source.index("_cast(network, residual, trt.float32)")
    bf16_residual_publish = residual_epilogue_source.rindex("trt.bfloat16")
    assert bias_add < scale_mul < residual_add < bf16_residual_publish

    occlusion_source = functions["_add_occlusion_embedding"]
    assert occlusion_source is not None
    memory_fp32 = occlusion_source.index("_cast(network, memory, trt.float32)")
    fp32_add = occlusion_source.index("memory_fp32,\n        occlusion")
    bf16_publish = occlusion_source.index("_cast(network, with_occlusion, trt.bfloat16)")
    assert memory_fp32 < fp32_add < bf16_publish


def test_sam3_tracker_attention_matches_torch_compile_precision_schedule() -> None:
    attention_path = Path(tracker_builder.__file__).with_name("tracker_attention_builder.py")
    module_source = attention_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(attention_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    linear_source = functions["_bf16_linear"]
    assert linear_source is not None
    bf16_input = linear_source.index("_cast(network, inp, trt.bfloat16)")
    matmul = linear_source.index("add_matmul_rhs_constant")
    bias = linear_source.index("add_bias_sum")
    assert bf16_input < matmul < bias
    assert "trt.float16" not in linear_source

    attention_source = functions["_attention_context"]
    assert attention_source is not None
    assert attention_source.count("trt.float16") == 3
    assert "add_attention_core(network, query, key, value)" in attention_source
    assert "trt.bfloat16" not in attention_source
    assert "trt.float32" not in attention_source

    for function_name in ("_self_attention", "_cross_attention"):
        source = functions[function_name]
        assert source is not None
        assert source.rindex("_bf16_linear(") < source.rindex("return projected")
        assert "_cast(network, projected, trt.float32)" not in source

    feed_forward_source = functions["_feed_forward"]
    assert feed_forward_source is not None
    assert feed_forward_source.count("_bf16_linear(") == 2
    assert "ActivationType.RELU" in feed_forward_source
    assert "return output" in feed_forward_source
    assert "_cast(network, output, trt.float32)" not in feed_forward_source


def test_sam3_tracker_attention_keeps_fp32_norm_rope_and_bf16_residual() -> None:
    attention_path = Path(tracker_builder.__file__).with_name("tracker_attention_builder.py")
    module_source = attention_path.read_text(encoding="utf-8")
    tree = ast.parse(module_source, filename=str(attention_path))
    functions = {
        node.name: ast.get_source_segment(module_source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    norm_source = functions["_layer_norm"]
    assert norm_source is not None
    norm_cast = norm_source.index("_cast(network, inp, trt.float32)")
    native_norm = norm_source.index("add_layer_norm_native")
    fp32_parameters = norm_source.index("dtype=np.float32")
    assert norm_cast < native_norm < fp32_parameters

    rope_source = functions["_apply_axial_rope"]
    assert rope_source is not None
    rope_cast = rope_source.index("_cast(network, inp, trt.float32)")
    rope_gather = rope_source.index("network.add_gather")
    rope_products = rope_source.index("ElementWiseOperation.PROD")
    assert rope_cast < rope_gather < rope_products

    sum_source = functions["_fp32_sum"]
    assert sum_source is not None
    assert sum_source.count("trt.float32") == 2
    assert "ElementWiseOperation.SUM" in sum_source

    spatial_memory_source = functions["_prepare_spatial_memory"]
    assert spatial_memory_source is not None
    assert "position_by_frame = memory_position" in spatial_memory_source
    assert "position_by_frame = _fp32_sum(" in spatial_memory_source
    assert "position_by_frame,\n        temporal_position" in spatial_memory_source

    residual_source = functions["_bf16_sum"]
    assert residual_source is not None
    assert residual_source.count("trt.bfloat16") == 2
    assert "ElementWiseOperation.SUM" in residual_source

    pointer_source = functions["_prepare_pointer_memory"]
    assert pointer_source is not None
    assert "_bf16_linear(" in pointer_source
    assert "_cast(network, projected_position, trt.float32)" not in pointer_source

    cross_attention_source = functions["_cross_attention"]
    assert cross_attention_source is not None
    assert "pointer_key_input = _bf16_sum(" in cross_attention_source

    public_source = functions["add_tracker_recurrent_conditioning"]
    assert public_source is not None
    assert public_source.count("_bf16_sum(") == 4
    assert "_cast(network, output, trt.bfloat16)" in public_source
    assert "_cast(network, position, trt.bfloat16)" in public_source
    final_norm = public_source.rindex("_layer_norm(")
    fp32_output = public_source.rindex("_tokens_to_features")
    assert final_norm < fp32_output
    assert "compute_precision" not in public_source
    assert "os.environ" not in module_source


def test_sam3_tracker_builder_has_only_required_direct_b1_b2_plans() -> None:
    parameters = inspect.signature(tracker_builder.build_sam3_tracker_engines).parameters
    assert tuple(parameters) == ("model_dir", "verbose")
    assert parameters["verbose"].default is False

    source = inspect.getsource(tracker_builder.build_sam3_tracker_engines)
    for section in (
        "TRACKER_INIT_SECTION",
        "TRACKER_STEP_SECTION",
        "TRACKER_STEP_BATCH2_SECTION",
        "TRACKER_MEMORY_SECTION",
        "TRACKER_MEMORY_BATCH2_SECTION",
        "TRACKER_HARD_MEMORY_SECTION",
        "TRACKER_HARD_MEMORY_BATCH2_SECTION",
        "HARD_MASK_RESIZE_SECTION",
        "HARD_MASK_RESIZE_BATCH2_SECTION",
    ):
        assert section in source

    assert source.count("_build_init(") == 1
    assert source.count("_build_step(") == 2
    assert source.count("_build_memory(") == 2
    assert source.count("_build_hard_memory(") == 2
    assert source.count("_build_hard_mask_resize(") == 2
    assert source.count("batch_size=1") == 4
    assert source.count("batch_size=2") == 4
    for forbidden in ("ffi", "aoti", ".pt2", "libtorch", "tvm"):
        assert forbidden not in source.lower()


def test_sam3_production_build_path_has_no_aoti_or_bridge_modules() -> None:
    family_dir = Path(tracker_builder.__file__).resolve().parent
    model_source = (family_dir / "model.py").read_text(encoding="utf-8")
    build_start = model_source.index("    def build_extra_engines(")
    build_end = model_source.index("\n    def get_segmentation_config", build_start)
    build_source = model_source[build_start:build_end]

    assert "build_sam3_tracker_engines" in build_source
    for forbidden in ("aoti", ".pt2", "libtorch", "tvm", "ffi", "native_plugin"):
        assert forbidden not in build_source.lower()

    for removed_module in (
        "hard_mask_resize_ffi_builder.py",
        "native_plugin_builder.py",
        "tracker_memory_ffi_builder.py",
        "tracker_step_ffi_builder.py",
    ):
        assert not (family_dir / removed_module).exists()

    repository_root = Path(__file__).resolve().parents[4]
    runtime_dir = repository_root / "src/runtime/models/sam3"
    assert not (runtime_dir / "sam3_tracker_step_runtime.cpp").exists()
    assert not (runtime_dir / "sam3_tracker_step_runtime.h").exists()


def test_sam3_tracker_build_needs_no_exported_runtime_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = object()
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(tracker_builder, "_read_model_config", lambda _: {})
    monkeypatch.setattr(tracker_builder, "_validate_video_policy", lambda _: None)
    monkeypatch.setattr(tracker_builder, "load_tracker_weights", lambda _: weights)
    monkeypatch.setattr(
        tracker_builder,
        "_build_init",
        lambda actual_weights, *, verbose: (
            events.append(("init", actual_weights, verbose)) or b"init"
        ),
    )

    def _build_fixed_batch(
        kind: str,
        actual_weights: object,
        *,
        batch_size: int,
        verbose: bool,
    ) -> bytes:
        events.append((kind, actual_weights, batch_size, verbose))
        return f"{kind}-b{batch_size}".encode()

    monkeypatch.setattr(
        tracker_builder,
        "_build_step",
        lambda actual_weights, *, batch_size, verbose: _build_fixed_batch(
            "step", actual_weights, batch_size=batch_size, verbose=verbose
        ),
    )
    monkeypatch.setattr(
        tracker_builder,
        "_build_memory",
        lambda actual_weights, *, batch_size, verbose: _build_fixed_batch(
            "memory", actual_weights, batch_size=batch_size, verbose=verbose
        ),
    )
    monkeypatch.setattr(
        tracker_builder,
        "_build_hard_memory",
        lambda actual_weights, *, batch_size, verbose: _build_fixed_batch(
            "hard-memory", actual_weights, batch_size=batch_size, verbose=verbose
        ),
    )

    def _build_resize(*, batch_size: int, verbose: bool) -> bytes:
        events.append(("hard-resize", batch_size, verbose))
        return f"hard-resize-b{batch_size}".encode()

    monkeypatch.setattr(tracker_builder, "_build_hard_mask_resize", _build_resize)

    plans = tracker_builder.build_sam3_tracker_engines(str(tmp_path), verbose=True)

    assert events == [
        ("init", weights, True),
        ("step", weights, 1, True),
        ("step", weights, 2, True),
        ("memory", weights, 1, True),
        ("memory", weights, 2, True),
        ("hard-memory", weights, 1, True),
        ("hard-memory", weights, 2, True),
        ("hard-resize", 1, True),
        ("hard-resize", 2, True),
    ]
    assert plans == {
        tracker_builder.TRACKER_INIT_SECTION: b"init",
        tracker_builder.TRACKER_STEP_SECTION: b"step-b1",
        tracker_builder.TRACKER_STEP_BATCH2_SECTION: b"step-b2",
        tracker_builder.TRACKER_MEMORY_SECTION: b"memory-b1",
        tracker_builder.TRACKER_MEMORY_BATCH2_SECTION: b"memory-b2",
        tracker_builder.TRACKER_HARD_MEMORY_SECTION: b"hard-memory-b1",
        tracker_builder.TRACKER_HARD_MEMORY_BATCH2_SECTION: b"hard-memory-b2",
        tracker_builder.HARD_MASK_RESIZE_SECTION: b"hard-resize-b1",
        tracker_builder.HARD_MASK_RESIZE_BATCH2_SECTION: b"hard-resize-b2",
    }


def test_sam3_runtime_loads_required_step_plans_without_external_bridge() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    plugin_path = repository_root / "src/runtime/models/sam3/plugin.cpp"
    source = plugin_path.read_text(encoding="utf-8")
    helper_start = source.index("Sam3TrackerStepModules load_sam3_tracker_step_modules")
    helper_end = source.index("\n}\n\n} // namespace", helper_start)
    helper = source[helper_start:helper_end]

    b1_load = helper.index('find_section(ctx.bundle, "sam3_tracker_step_engine_plan")')
    b2_load = helper.index('find_section(ctx.bundle, "sam3_tracker_step_batch2_engine_plan")')
    assert b1_load < b2_load
    assert helper.count("load_trt_module_from_plan(") == 2
    assert "extract_optional_module" not in helper
    assert "if (!video_tracking_supported)" in helper
    for forbidden in ("aoti", ".pt2", "libtorch", "torch", "tvm", "ffi", "dlopen"):
        assert forbidden not in helper.lower()

    plugin_helpers_path = repository_root / "src/runtime/models/sam3/plugin_helpers.cpp"
    production_loader_source = source + plugin_helpers_path.read_text(encoding="utf-8")
    for forbidden in (
        "load_sam3_tracker_step_runtime",
        "load_ffi_kernels_from_bundle",
        "sam3_tracker_aoti",
        ".pt2",
        "libtorch",
        "libtvm_ffi",
    ):
        assert forbidden not in production_loader_source


def test_sam3_runtime_requires_all_memory_plans_only_for_video_bundles() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    plugin_path = repository_root / "src/runtime/models/sam3/plugin.cpp"
    source = plugin_path.read_text(encoding="utf-8")
    helper_start = source.index("Sam3TrackerMemoryModules load_sam3_tracker_memory_modules")
    helper_end = source.index("\n}\n\nSam3HardMaskResizeModules", helper_start)
    helper = source[helper_start:helper_end]

    assert "if (!video_tracking_supported)" in helper
    assert helper.count("load_trt_module_from_plan(") == 4
    assert "extract_optional_module" not in helper
    for section in (
        "sam3_tracker_memory_engine_plan",
        "sam3_tracker_memory_batch2_engine_plan",
        "sam3_tracker_hard_memory_engine_plan",
        "sam3_tracker_hard_memory_batch2_engine_plan",
    ):
        assert f'find_section(ctx.bundle, "{section}")' in helper

    resize_start = source.index("Sam3HardMaskResizeModules load_sam3_hard_mask_resize_modules")
    resize_end = source.index("\n}\n\nSam3TrackerStepModules", resize_start)
    resize_helper = source[resize_start:resize_end]
    assert "if (!video_tracking_supported)" in resize_helper
    assert resize_helper.count("load_trt_module_from_plan(") == 2
    assert "extract_optional_module" not in resize_helper
    for section in (
        "sam3_hard_mask_resize_engine_plan",
        "sam3_hard_mask_resize_batch2_engine_plan",
    ):
        assert f'find_section(ctx.bundle, "{section}")' in resize_helper

    create_start = source.index("std::unique_ptr<IPipeline> create")
    create_end = source.index("\n    }\n};", create_start)
    create = source[create_start:create_end]
    assert create.index("load_sam3_tracker_step_modules") < create.index(
        "load_sam3_tracker_memory_modules"
    )


def test_sam3_tracker_reviewed_video_bound_derives_pointer_profile() -> None:
    assert tracker_builder.SAM3_TRACKER_MAX_VIDEO_FRAMES == 1024
    assert tracker_builder.SAM3_TRACKER_RECONDITION_CADENCE == 16
    assert tracker_builder.SAM3_TRACKER_MAX_CONDITIONING_POINTERS == 4
    assert tracker_builder.SAM3_TRACKER_MAX_POINTER_INPUTS == 19


def _sam3_production_sources() -> dict[Path, str]:
    family_dir = Path(tracker_builder.__file__).resolve().parent
    sources = {path: path.read_text(encoding="utf-8") for path in sorted(family_dir.glob("*.py"))}
    assert sources, "SAM3 production sources were not found"
    return sources


def _call_attribute_names(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_sam3_production_has_no_exchange_graph_path() -> None:
    """Forbid exchange graphs and framework exporters in production SAM3."""

    forbidden_import_roots = {
        "onnx",
        "onnx_ir",
        "onnxscript",
    }
    tracker_framework_roots = {"torch", "transformers"}
    forbidden_source_tokens = {
        ".onnx",
        ".pt2",
        "aoti",
        "libtorch",
        "modelproto",
        "nvonnxparser",
        "onnxparser",
        "opset",
        "torch.export",
        "torch.onnx",
        "tvm_ffi",
    }

    for path, source in _sam3_production_sources().items():
        tree = ast.parse(source, filename=str(path))
        imports: set[str] = set()
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", maxsplit=1)[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name.lower())
            elif isinstance(node, ast.Attribute):
                symbols.add(node.attr.lower())
            elif isinstance(node, ast.Name):
                symbols.add(node.id.lower())

        assert not (imports & forbidden_import_roots), (
            f"{path.name} imports an exchange-graph dependency: "
            f"{sorted(imports & forbidden_import_roots)}"
        )
        if path.name.startswith("tracker"):
            assert not (imports & tracker_framework_roots), (
                f"{path.name} imports a framework dependency: "
                f"{sorted(imports & tracker_framework_roots)}"
            )
        lowered = source.lower()
        matched_tokens = sorted(token for token in forbidden_source_tokens if token in lowered)
        assert not matched_tokens, (
            f"{path.name} contains a forbidden exchange-graph path: {matched_tokens}"
        )
        forbidden_symbols = sorted(
            symbol
            for symbol in symbols
            if "onnx" in symbol
            or "opset" in symbol
            or "modelproto" in symbol
            or "export" in symbol
            or symbol in {"parse", "parse_from_file"}
            or symbol.endswith("parser")
        )
        assert not forbidden_symbols, (
            f"{path.name} defines or calls a graph exporter/parser: {forbidden_symbols}"
        )

    family_dir = Path(tracker_builder.__file__).resolve().parent
    assert not list(family_dir.glob("*_aoti_exporter.py"))


def test_sam3_family_has_no_embedded_test_sources() -> None:
    family_dir = Path(tracker_builder.__file__).resolve().parent
    assert not list((family_dir / "tests").rglob("*.py"))


def test_sam3_tracker_graph_is_built_with_native_tensorrt_api() -> None:
    """Guard the build boundary, not just the absence of a particular parser."""

    calls: set[str] = set()
    for path, source in _sam3_production_sources().items():
        if not path.name.startswith("tracker"):
            continue
        calls.update(_call_attribute_names(ast.parse(source, filename=str(path))))

    assert {"Builder", "create_network", "add_input", "build_serialized_network"} <= calls
    native_layer_calls = {
        "add_activation",
        "add_attention",
        "add_convolution_nd",
        "add_elementwise",
        "add_matrix_multiply",
        "add_resize",
        "add_shuffle",
        "add_slice",
        "add_topk",
    }
    used_native_layers = calls & native_layer_calls
    assert len(used_native_layers) >= 4, (
        "SAM3 tracker sources must reconstruct the graph with TensorRT layers; "
        f"found only {sorted(used_native_layers)}"
    )


def test_sam3_graph_ops_has_no_external_kernel_bridge() -> None:
    graph_ops_path = Path(tracker_builder.__file__).with_name("graph_ops.py")
    source = graph_ops_path.read_text(encoding="utf-8")
    for forbidden in (
        "add_decoder_attention_ffi",
        "add_tvm_ffi_kernel",
        "TvmFfiKernel",
        "get_plugin_registry",
        "add_plugin_v2",
        "onnx",
    ):
        assert forbidden not in source


def test_sam3_tracker_video_policy_rejects_unreviewed_cadence(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"recondition_every_nth_frame": 16}), encoding="utf-8")
    tracker_builder._validate_video_policy(str(tmp_path))

    config_path.write_text(json.dumps({"recondition_every_nth_frame": 8}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reviewed reconditioning cadence 16"):
        tracker_builder._validate_video_policy(str(tmp_path))


class _FakeSafeTensorReader:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self.tensors = tensors
        self.reads: list[str] = []

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key: str) -> np.ndarray:
        self.reads.append(key)
        return self.tensors[key]


def test_sam3_tracker_weights_are_loaded_directly_as_numpy(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    reader = _FakeSafeTensorReader(
        {
            "tracker_model.projection.weight": np.arange(6, dtype=np.float16).reshape(2, 3),
            "tracker_model.projection.bias": np.array([2.0, 3.0], dtype=np.float16),
            "vision_encoder.unrelated": np.ones((1,), dtype=np.float32),
        }
    )
    opened: list[tuple[str, str]] = []

    def fake_safe_open(path: str, *, framework: str):
        opened.append((path, framework))
        return reader

    monkeypatch.setattr(tracker_weights, "safe_open", fake_safe_open)
    weights = tracker_weights.load_tracker_weights(tmp_path)

    np.testing.assert_array_equal(
        weights.linear_weight("projection"),
        np.arange(6, dtype=np.float32).reshape(2, 3).T,
    )
    np.testing.assert_array_equal(
        weights.linear_bias("projection"), np.array([2.0, 3.0], dtype=np.float32)
    )
    assert weights["projection.bias"].flags.c_contiguous
    assert opened == [(str(checkpoint), "numpy")]
    assert reader.reads.count("tracker_model.projection.bias") == 1
    with pytest.raises(KeyError, match="Missing SAM3 tracker parameter"):
        _ = weights["vision_encoder.unrelated"]


def test_sam3_tracker_weights_load_only_tracker_shards(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "tracker_model.a.weight": "tracker-1.safetensors",
                    "tracker_model.b.weight": "tracker-2.safetensors",
                    "vision_encoder.weight": "vision.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    readers = {
        "tracker-1.safetensors": _FakeSafeTensorReader(
            {"tracker_model.a.weight": np.ones((1,), dtype=np.float32)}
        ),
        "tracker-2.safetensors": _FakeSafeTensorReader(
            {"tracker_model.b.weight": np.full((1,), 2.0, dtype=np.float32)}
        ),
    }
    opened: list[str] = []

    def fake_safe_open(path: str, *, framework: str):
        assert framework == "numpy"
        name = Path(path).name
        opened.append(name)
        return readers[name]

    monkeypatch.setattr(tracker_weights, "safe_open", fake_safe_open)
    weights = tracker_weights.load_tracker_weights(tmp_path)

    assert sorted(opened) == ["tracker-1.safetensors", "tracker-2.safetensors"]
    np.testing.assert_array_equal(weights["a.weight"], np.ones((1,), dtype=np.float32))
    np.testing.assert_array_equal(weights["b.weight"], np.full((1,), 2.0, dtype=np.float32))
