# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ref2VA's native timestep projection and original mixed-precision AdaLN."""

from dataclasses import replace
import json

import pytest

from families.minimax_h3.adaln_builder import build_adaln_precompute_engine
from families.minimax_h3.config import SOL_ENGINE_1344X768_124F


@pytest.mark.parametrize("mode", [None, 1, "yes"])
def test_reference_adaln_rejects_non_boolean_input_mode_before_cuda(mode):
    with pytest.raises(ValueError, match="timestep projection"):
        build_adaln_precompute_engine({}, SOL_ENGINE_1344X768_124F, input_timesteps=mode)


def test_reference_adaln_rejects_odd_frequency_width_before_cuda():
    with pytest.raises(ValueError, match="even feature width"):
        build_adaln_precompute_engine(
            {}, replace(SOL_ENGINE_1344X768_124F, timestep_input_dim=3), input_timesteps=True
        )


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("frequency_width", [4, 256])
def test_reference_adaln_matches_original_timestep_and_modulation(
    frequency_width, record_property, tmp_path
):
    import importlib.metadata
    import tensorrt as trt
    import torch
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation
    from .test_trt_builders import _weights

    assert importlib.metadata.version("diffusers") == "0.40.0"
    profile = replace(
        SOL_ENGINE_1344X768_124F,
        hidden_size=8,
        num_layers=2,
        timestep_input_dim=frequency_width,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
    )
    weights = _weights(profile)
    projection = Timesteps(frequency_width, flip_sin_to_cos=True, downscale_freq_shift=0).cuda()
    embedder = TimestepEmbedding(frequency_width, 8, out_dim=4).float().cuda()
    embedder.load_state_dict(
        {
            name.removeprefix("time_embedder."): torch.from_numpy(value)
            for name, value in weights.items()
            if name.startswith("time_embedder.")
        }
    )
    blocks = []
    for index in range(2):
        block = MiniMaxH3AdaLayerNormModulation(4, 8).bfloat16().cuda()
        prefix = f"transformer_blocks.{index}.adaln_proj."
        block.load_state_dict(
            {
                name.removeprefix(prefix): torch.from_numpy(value)
                for name, value in weights.items()
                if name.startswith(prefix)
            }
        )
        blocks.append(block)
    final = torch.nn.Linear(4, 16).bfloat16().cuda()
    final.load_state_dict(
        {
            name.removeprefix("norm_out.linear."): torch.from_numpy(value)
            for name, value in weights.items()
            if name.startswith("norm_out.linear.")
        }
    )
    plan = build_adaln_precompute_engine(
        weights, profile, input_timesteps=True, workspace_bytes=1 << 30
    )
    (tmp_path / "adaln.plan").write_bytes(plan)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_io_tensors == 4
    assert engine.get_tensor_shape("timesteps") == (4,)
    context = engine.create_execution_context()
    assert context is not None
    inputs = torch.tensor([0.0, 0.12345, 0.999, 0.9995], device="cuda", dtype=torch.float32)
    inputs.cpu().numpy().tofile(tmp_path / "timesteps.f32.bin")
    actual = {
        f"block_modulation_{index}": torch.empty((12, 6, 8), device="cuda", dtype=torch.bfloat16)
        for index in range(2)
    }
    actual["final_modulation"] = torch.empty((4, 2, 8), device="cuda", dtype=torch.bfloat16)
    assert context.set_tensor_address("timesteps", inputs.data_ptr())
    for name, value in actual.items():
        assert context.set_tensor_address(name, value.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    with torch.no_grad():
        temb = embedder(projection(inputs))
        expected = {
            f"block_modulation_{index}": torch.stack(block(temb), dim=1)
            for index, block in enumerate(blocks)
        }
        expected["final_modulation"] = final(torch.nn.functional.silu(temb).bfloat16()).reshape(
            4, 2, 8
        )
    comparisons = []
    for name in actual:
        actual[name].view(torch.uint16).cpu().numpy().tofile(tmp_path / (name + ".actual.bf16.bin"))
        expected[name].contiguous().view(torch.uint16).cpu().numpy().tofile(
            tmp_path / (name + ".upstream.bf16.bin")
        )
        # The same predeclared BF16 criterion used by the family's other native
        # reference graph tests. Do not loosen it in response to a mismatch.
        torch.testing.assert_close(actual[name], expected[name], rtol=0.016, atol=1e-5)
        comparisons.append(
            dict(
                name=name,
                max_abs_error=(actual[name].float() - expected[name].float()).abs().max().item(),
            )
        )
    record_property(
        "reference_adaln",
        json.dumps(
            dict(
                frequency_width=frequency_width,
                comparisons=comparisons,
                trained_checkpoint_qualified=False,
                native_Ref2VA_video_qualified=False,
            )
        ),
    )
