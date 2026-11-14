# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise variable-size Qwen3-VL vision against the HF mathematical reference."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.qwen_vl.model import (  # noqa: E402
    _QwenVLModel,
    _default_sequence_length,
    _dynamic_vision_profile,
)


def test_dynamic_qwen3_preserves_processor_area_limits_and_runtime_contract(tmp_path):
    config = SimpleNamespace(raw={
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2,
                          "hidden_size": 1024, "num_heads": 16,
                          "num_position_embeddings": 2304,
                          "deepstack_visual_indexes": [5, 11, 17]},
        "_family_build_options": {"qwen_vl_vision": {"dynamic_resolution": True,
                                                     "max_pixels": 4194304}},
    }, hidden_size=2048)
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"shortest_edge": 65536, "longest_edge": 4194304}}))
    enabled, minimum, optimum, maximum = _dynamic_vision_profile(str(tmp_path), config)
    assert (enabled, minimum, optimum, maximum) == (True, 65536, 200704, 4194304)
    config.raw["_qwen_vl_dynamic_vision_profile"] = {
        "enabled": enabled, "min_pixels": minimum, "opt_pixels": optimum, "max_pixels": maximum}
    runtime = _QwenVLModel().get_vl_config(config)
    assert runtime["preprocessor_type"] == "qwen3_smart_resize_patchify"
    assert runtime["num_image_pad_tokens"] == 0
    assert runtime["vision_position_grid_size"] == 48
    assert runtime["max_pixels"] == maximum
    assert runtime["deepstack_num_levels"] == 3


def test_ordinary_qwen3_build_reserves_its_448_square_image_tokens(tmp_path):
    config = SimpleNamespace(raw={
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2,
                          "hidden_size": 1024, "num_heads": 16,
                          "num_position_embeddings": 2304,
                          "deepstack_visual_indexes": [5, 11, 17]},
    }, hidden_size=2048, max_position_embeddings=32768)
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"shortest_edge": 65536, "longest_edge": 4194304}}))
    assert _dynamic_vision_profile(str(tmp_path), config) == (True, 65536, 200704, 200704)
    assert _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    # The profile admits up to 198 merged tokens at a 200:1 aspect ratio, plus 256 text tokens.
    assert _default_sequence_length(str(tmp_path), config) == 454
    config.raw["_family_build_options"] = {"qwen_vl_vision": {"max_pixels": 4194304}}
    # 4,096 visual tokens at the checkpoint's own maximum area, plus 256 text tokens.
    assert _default_sequence_length(str(tmp_path), config) == 4352
    config.raw["_family_build_options"] = {"qwen_vl_vision": {"dynamic_resolution": False}}
    assert not _dynamic_vision_profile(str(tmp_path), config)[0]
    assert not _QwenVLModel().get_vl_config(config)["dynamic_image_resolution"]
    assert _default_sequence_length(str(tmp_path), config) == 256


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_dynamic_qwen3_main_and_deepstack_outputs_match_hf(precision):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    from families.qwen_vl.qwen_vl_vision_builder import build_qwen3_vl_vision_engine

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    config = Qwen3VLVisionConfig(
        depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
        patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
        num_position_embeddings=16, deepstack_visual_indexes=[0], out_hidden_size=32,
        hidden_act="gelu_pytorch_tanh")
    config._attn_implementation = "eager"
    torch.manual_seed(1731)
    reference = Qwen3VLVisionModel(config).eval().to(dtype=dtype, device="cuda")
    weights = {"visual." + name: value.detach().cpu().float().numpy()
               for name, value in reference.state_dict().items()}
    plan = build_qwen3_vl_vision_engine(
        config.to_dict(), weights, dynamic_image_resolution=True,
        min_image_pixels=16, opt_image_pixels=64, max_image_pixels=256, precision=precision)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    for height, width in [(2, 2), (2, 6), (8, 8), (6, 4), (2, 2)]:
        patches = height * width
        grid = torch.tensor([[1, height, width]], dtype=torch.int64, device="cuda")
        values = torch.randn((patches, 24), dtype=torch.float32, device="cuda")
        # Independent HF-5.2 interpolation contract, then merge-group order.
        rows = torch.linspace(0, 3, height, device="cpu")
        columns = torch.linspace(0, 3, width, device="cpu")
        r0, c0 = rows.int(), columns.int()
        r1, c1 = (r0 + 1).clip(max=3), (c0 + 1).clip(max=3)
        dr, dc = rows - r0, columns - c0
        indices = torch.stack([(r0[:, None] * 4 + c0).flatten(),
                               (r0[:, None] * 4 + c1).flatten(),
                               (r1[:, None] * 4 + c0).flatten(),
                               (r1[:, None] * 4 + c1).flatten()])
        blends = torch.stack([((1 - dr[:, None]) * (1 - dc)).flatten(),
                              ((1 - dr[:, None]) * dc).flatten(),
                              (dr[:, None] * (1 - dc)).flatten(),
                              (dr[:, None] * dc).flatten()])

        def grouped(tensor):
            return tensor.reshape(4, height // 2, 2, width // 2, 2).permute(0, 1, 3, 2, 4).flatten(1)

        with torch.inference_mode():
            expected = reference(values.to(dtype), grid_thw=grid)
            rope = reference.rot_pos_emb(grid).reshape(patches, -1)
        # Native half precision rounds RoPE multiplications back to Q/K dtype.
        inputs = {"pixel_values": values,
                  "vision_cos_half": rope.cos().float().contiguous(),
                  "vision_sin_half": rope.sin().float().contiguous()}
        grouped_indices = grouped(indices).to(device="cuda", dtype=torch.int32)
        grouped_blends = grouped(blends).cuda()
        for corner in range(4):
            inputs[f"vision_position_indices_{corner}"] = grouped_indices[corner].contiguous()
            inputs[f"vision_position_weights_{corner}"] = grouped_blends[corner].reshape(-1, 1).contiguous()
        for name, tensor in inputs.items():
            assert context.set_input_shape(name, tuple(tensor.shape))
            assert context.set_tensor_address(name, tensor.data_ptr())
        outputs = {name: torch.empty(tuple(context.get_tensor_shape(name)), device="cuda", dtype=torch.float32)
                   for name in ("image_features", "deepstack_features_0")}
        for name, tensor in outputs.items():
            assert context.set_tensor_address(name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        if hasattr(expected, "pooler_output"):
            main, deep = expected.pooler_output, expected.deepstack_features
        else:
            main, deep = expected
        tolerance = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 2e-2}[precision]
        np.testing.assert_allclose(outputs["image_features"].cpu().numpy(), main.float().cpu().numpy(),
                                   atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(outputs["deepstack_features_0"].cpu().numpy(), deep[0].float().cpu().numpy(),
                                   atol=tolerance, rtol=tolerance)
