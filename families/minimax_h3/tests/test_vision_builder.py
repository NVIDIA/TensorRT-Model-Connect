# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision graph contracts and independent pinned Qwen3-VL component oracles."""

from dataclasses import replace
import json

import ml_dtypes
import numpy as np
import pytest

from families.minimax_h3.checkpoint import numpy_state
from families.minimax_h3.vision_builder import (
    VisionConfig,
    _validate_weights,
    build_vision_encoder_engine,
    checkpoint_keys,
    checkpoint_shapes,
)


TINY = replace(
    VisionConfig(),
    depth=3,
    hidden_size=16,
    intermediate_size=32,
    num_heads=2,
    patch_size=2,
    out_hidden_size=8,
    num_position_embeddings=64,
    deepstack_visual_indexes=(0, 1, 2),
)


def test_original_vision_checkpoint_schema_covers_all_351_tensors():
    shapes = checkpoint_shapes()
    assert len(shapes) == len(checkpoint_keys()) == 351
    assert shapes["model.visual.patch_embed.proj.weight"] == (1152, 3, 2, 16, 16)
    assert shapes["model.visual.pos_embed.weight"] == (2304, 1152)
    assert shapes["model.visual.merger.norm.weight"] == (1152,)
    assert shapes["model.visual.deepstack_merger_list.0.norm.weight"] == (4608,)
    assert shapes["model.visual.deepstack_merger_list.2.linear_fc2.weight"] == (5120, 4608)
    assert not any("language_model" in name for name in shapes)


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_size": 17},
        {"hidden_size": 18},
        {"num_heads": 0},
        {"patch_size": True},
        {"depth": 2},
        {"num_position_embeddings": 63},
        {"deepstack_visual_indexes": (0, 0, 1)},
        {"deepstack_visual_indexes": (0, 2, 1)},
        {"deepstack_visual_indexes": (0, 1)},
        {"deepstack_visual_indexes": (False, 1, 2)},
    ],
)
def test_vision_config_fails_closed(change):
    with pytest.raises(ValueError):
        checkpoint_shapes(replace(TINY, **change))


@pytest.mark.parametrize(
    "rows", [(1, 4, 8), (4, 8), (4, 7, 8), (8, 4, 16), (True, 4, 8), (4, 8, 16.0)]
)
def test_vision_profiles_reject_invalid_patch_counts_before_cuda(rows):
    with pytest.raises(ValueError, match="merge-aligned"):
        build_vision_encoder_engine({}, config=TINY, patch_rows=rows)


def test_vision_weights_never_silently_expand_or_downcast_checkpoint_storage():
    weights = {
        name: np.zeros(shape, dtype=ml_dtypes.bfloat16)
        for name, shape in checkpoint_shapes(TINY).items()
    }
    _validate_weights(weights, TINY, "bf16")
    with pytest.raises(ValueError, match="requires float32"):
        _validate_weights(weights, TINY, "fp32")
    with pytest.raises(ValueError, match="exactly"):
        _validate_weights({**weights, "model.language_model.invalid": np.zeros(1)}, TINY, "bf16")
    with pytest.raises(ValueError, match="original BF16"):
        _validate_weights(weights, TINY, "fp16")


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_reference_vision_matches_upstream_all_deepstack_outputs(precision, record_property):
    import tensorrt as trt
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    torch.manual_seed(431)
    config = Qwen3VLVisionConfig(**TINY.__dict__, hidden_act="gelu_pytorch_tanh")
    # The pinned conditioner supports/defaults to SDPA, not an eager BF16
    # score tensor with an additional intermediate rounding boundary.
    config._attn_implementation = "sdpa"
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    original = Qwen3VLVisionModel(config).eval()
    rotary_frequency = original.rotary_pos_emb.inv_freq.clone()
    original.to(dtype=dtype)
    # A dtype-changing Module.to also casts nonpersistent buffers. Restore the
    # explicitly FP32 frequency buffer used by from_pretrained, before CUDA.
    original.rotary_pos_emb.inv_freq = rotary_frequency
    weights = numpy_state(
        {f"model.visual.{name}": value for name, value in original.state_dict().items()}
    )
    assert {name: value.shape for name, value in weights.items()} == checkpoint_shapes(TINY)
    plan = build_vision_encoder_engine(
        weights, config=TINY, patch_rows=(4, 8, 24), precision=precision, workspace_bytes=1 << 30
    )
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_io_tensors == 9
    context = engine.create_execution_context()
    assert context is not None
    original.cuda()
    stream = torch.cuda.current_stream()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            for height, width in ((2, 2), (2, 4), (4, 6)):
                rows = height * width
                grid = torch.tensor([[1, height, width]], device="cuda")
                patches = torch.randn(rows, TINY.patch_dim, device="cuda", dtype=torch.float32)
                indices, scales = get_vision_interpolation_indices_and_weights(
                    grid, 8, mode="bilinear", align_corners=True, spatial_merge_size=2
                )
                positions = get_vision_position_ids(grid, 2)
                frequencies = original.rotary_pos_emb(positions).reshape(rows, -1)
                frequencies = torch.cat((frequencies, frequencies), dim=-1)
                inputs = dict(
                    pixel_values=patches,
                    position_indices=indices.int(),
                    position_weights=scales,
                    rotary_cos=frequencies.cos(),
                    rotary_sin=frequencies.sin(),
                )
                expected = original(patches.to(dtype), grid_thw=grid, return_dict=True)
                expected_values = dict(
                    visual_features=expected.pooler_output,
                    **{
                        f"deepstack_features_{i}": value
                        for i, value in enumerate(expected.deepstack_features)
                    },
                )
                for name, value in inputs.items():
                    assert context.set_input_shape(name, tuple(value.shape))
                    assert context.set_tensor_address(name, value.data_ptr())
                outputs = {}
                for name in expected_values:
                    shape = tuple(context.get_tensor_shape(name))
                    assert shape == (rows // 4, TINY.out_hidden_size)
                    output = torch.empty(shape, device="cuda", dtype=dtype)
                    outputs[name] = output
                    assert context.set_tensor_address(name, output.data_ptr())
                assert context.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
                for name, actual in outputs.items():
                    expected_value = expected_values[name]
                    error = (actual.float() - expected_value.float()).abs()
                    results.append(
                        dict(grid=[height, width], output=name, max_abs_error=error.max().item())
                    )
                    # Frozen PyTorch defaults for each native storage dtype.
                    rtol, atol = (1.3e-6, 1e-5) if precision == "fp32" else (0.016, 1e-5)
                    torch.testing.assert_close(actual, expected_value, rtol=rtol, atol=atol)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    record_property("vision_component_comparisons", json.dumps(results, allow_nan=False))
