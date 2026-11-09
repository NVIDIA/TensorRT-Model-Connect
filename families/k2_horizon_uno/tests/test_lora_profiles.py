# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LoRA profile applies the conditional branch; the base profile skips it."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.k2_horizon_uno import model  # noqa: E402


_VOCAB, _IN, _OUT, _RANK, _SCALE, _ROWS = 16, 32, 48, 4, 2.0, 8


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16)
    return np.ascontiguousarray(tensor.view(torch.int16).numpy().view(np.uint16))


def _from_bits(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).float()


def _build_projection_engine(table, weight, lora_a, lora_b):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    token_id = network.add_input("token_id", trt.int32, (-1,))
    lora_mask = network.add_input("lora_mask", trt.float32, (-1,))
    model.add_lora_and_base_profiles(builder, config, ("token_id",), _ROWS)
    keepalive: list[np.ndarray] = []
    embedding = model._work_constant(
        network, table.shape, table, work_dtype=trt.bfloat16, constant_keepalive=keepalive
    )
    rows = network.add_gather(embedding, token_id, 0).get_output(0)
    output = model._conditional_projection(
        network,
        rows,
        weight,
        lora_a,
        lora_b,
        lora_mask,
        model._lora_rows(network, token_id, lora_mask),
        lhs_width=_IN,
        rhs_width=_OUT,
        rank=_RANK,
        scale=_SCALE,
        work_dtype=trt.bfloat16,
        constant_keepalive=keepalive,
        name="probe",
    )
    output = model._cast(network, output, trt.float32)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    assert engine.num_optimization_profiles == 2
    return engine


def _run(engine, profile: int, tokens: list[int], mask: list[float]) -> torch.Tensor:
    stream = torch.cuda.Stream()
    context = engine.create_execution_context()
    context.set_optimization_profile_async(profile, stream.cuda_stream)
    token_tensor = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    # An empty mask still needs a valid device address.
    mask_tensor = torch.tensor([*mask, 0.0], dtype=torch.float32, device="cuda")
    output = torch.full((len(tokens), _OUT), float("nan"), device="cuda")
    context.set_input_shape("token_id", (len(tokens),))
    context.set_input_shape("lora_mask", (len(mask),))
    context.set_tensor_address("token_id", token_tensor.data_ptr())
    context.set_tensor_address("lora_mask", mask_tensor.data_ptr())
    context.set_tensor_address("output", output.data_ptr())
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return output.cpu()


@pytest.mark.gpu
@pytest.mark.trt
def test_base_profile_skips_lora_and_lora_profile_applies_the_mask():
    rng = np.random.default_rng(0)
    table = _bf16_bits(rng.standard_normal((_VOCAB, _IN)))
    weight = _bf16_bits(rng.standard_normal((_IN, _OUT)) * 0.1)
    lora_a = _bf16_bits(rng.standard_normal((_IN, _RANK)) * 0.1)
    lora_b = _bf16_bits(rng.standard_normal((_RANK, _OUT)) * 0.1)
    engine = _build_projection_engine(table, weight, lora_a, lora_b)

    tokens = [3, 1, 4, 1, 5, 9, 2, 6]
    x = _from_bits(table)[tokens]
    base = x @ _from_bits(weight)
    delta = (x @ _from_bits(lora_a)) @ _from_bits(lora_b) * _SCALE

    draft_mask = [0.0] + [1.0] * 7
    lora = _run(engine, model.LORA_PROFILE, tokens, draft_mask)
    expected = base + delta * torch.tensor(draft_mask)[:, None]
    torch.testing.assert_close(lora, expected, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(lora[1:], base[1:], rtol=2e-2, atol=2e-2)

    zero_mask = _run(engine, model.LORA_PROFILE, tokens, [0.0] * 8)
    for rows in (8, 3, 1):
        skipped = _run(engine, model.BASE_PROFILE, tokens[:rows], [])
        torch.testing.assert_close(skipped, base[:rows], rtol=2e-2, atol=2e-2)
        if rows == 8:
            assert torch.equal(skipped, zero_mask)
