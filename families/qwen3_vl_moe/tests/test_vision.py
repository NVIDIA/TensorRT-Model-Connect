# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Qwen3-VL MoE vision and DeepStack reference checks."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_dynamic_qwen3_main_and_deepstack_outputs_match_hf(precision):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

    from families.qwen3_vl_moe.qwen3_vl_moe_vision_builder import build_qwen3_vl_vision_engine

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    config = Qwen3VLMoeVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        out_hidden_size=32,
        hidden_act="gelu_pytorch_tanh",
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1731)
    reference = Qwen3VLMoeVisionModel(config).eval().to(dtype=dtype, device="cuda")
    weights = {
        "visual." + name: value.detach().cpu().float().numpy()
        for name, value in reference.state_dict().items()
    }
    plan = build_qwen3_vl_vision_engine(
        config.to_dict(),
        weights,
        dynamic_image_resolution=True,
        min_image_pixels=16,
        opt_image_pixels=64,
        max_image_pixels=256,
        precision=precision,
    )
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
        indices = torch.stack(
            [
                (r0[:, None] * 4 + c0).flatten(),
                (r0[:, None] * 4 + c1).flatten(),
                (r1[:, None] * 4 + c0).flatten(),
                (r1[:, None] * 4 + c1).flatten(),
            ]
        )
        blends = torch.stack(
            [
                ((1 - dr[:, None]) * (1 - dc)).flatten(),
                ((1 - dr[:, None]) * dc).flatten(),
                (dr[:, None] * (1 - dc)).flatten(),
                (dr[:, None] * dc).flatten(),
            ]
        )

        def grouped(tensor):
            return (
                tensor.reshape(4, height // 2, 2, width // 2, 2).permute(0, 1, 3, 2, 4).flatten(1)
            )

        with torch.inference_mode():
            expected = reference(values.to(dtype), grid_thw=grid)
            rope = reference.rot_pos_emb(grid).reshape(patches, -1)
        # Native half precision rounds RoPE multiplications back to Q/K dtype.
        inputs = {
            "pixel_values": values,
            "vision_cos_half": rope.cos().float().contiguous(),
            "vision_sin_half": rope.sin().float().contiguous(),
        }
        grouped_indices = grouped(indices).to(device="cuda", dtype=torch.int32)
        grouped_blends = grouped(blends).cuda()
        for corner in range(4):
            inputs[f"vision_position_indices_{corner}"] = grouped_indices[corner].contiguous()
            inputs[f"vision_position_weights_{corner}"] = (
                grouped_blends[corner].reshape(-1, 1).contiguous()
            )
        for name, tensor in inputs.items():
            assert context.set_input_shape(name, tuple(tensor.shape))
            assert context.set_tensor_address(name, tensor.data_ptr())
        outputs = {
            name: torch.empty(
                tuple(context.get_tensor_shape(name)), device="cuda", dtype=torch.float32
            )
            for name in ("image_features", "deepstack_features_0")
        }
        for name, tensor in outputs.items():
            assert context.set_tensor_address(name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        if hasattr(expected, "pooler_output"):
            main, deep = expected.pooler_output, expected.deepstack_features
        else:
            main, deep = expected
        tolerance = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 2e-2}[precision]
        np.testing.assert_allclose(
            outputs["image_features"].cpu().numpy(),
            main.float().cpu().numpy(),
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_allclose(
            outputs["deepstack_features_0"].cpu().numpy(),
            deep[0].float().cpu().numpy(),
            atol=tolerance,
            rtol=tolerance,
        )
