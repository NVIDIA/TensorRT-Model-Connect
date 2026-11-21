# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU regression candidate: packed LongRoPE tables versus independent eager math."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

from families.phi4_multimodal import graph_ops  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.trt,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]

RAW = {"original_max_position_embeddings": 4096, "max_position_embeddings": 131072,
       "rope_scaling": {"type": "longrope", "short_factor": [1.0] * 48,
                        "long_factor": np.linspace(1.0, 8.0, 48).tolist()}}


@pytest.fixture(scope="module")
def engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.float16, (3, 256))
    positions = network.add_input("position_id", trt.int32, (3,))
    cosine, sine, threshold = graph_ops.make_rope_half_tables(RAW, 4098, 128, 10000.0, .75)
    assert threshold == 4096
    adjusted_positions = graph_ops.add_rope_regime_positions(network, positions, 4098)
    caches = [graph_ops.add_constant(network, table.shape, table.astype(np.float16), dtype=np.float16)
              for table in (cosine, sine)]
    output = graph_ops.add_apply_rope_native(network, x, 2, 128, *caches,
                                            adjusted_positions, 96, sequence_length=3)
    output.name = "y"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    yield engine


@pytest.mark.parametrize("sequence_length", [512, 4096, 4097])
def test_longrope_uses_actual_sequence_regime(engine, sequence_length):
    torch.manual_seed(1774)
    x = torch.randn(3, 2, 128, device="cuda", dtype=torch.float16)
    positions = torch.tensor([0, 127, 511], device="cuda", dtype=torch.int32)
    long_factor = sequence_length > 4096
    regime = torch.tensor([int(long_factor)], device="cuda", dtype=torch.int32)

    # Independent checkpoint equation: FP32 frequencies and trigonometry,
    # then FP16 cos/sin and FP16 rotate-half arithmetic.
    factors = torch.tensor(RAW["rope_scaling"]["long_factor" if long_factor else "short_factor"],
                           device="cuda", dtype=torch.float32)
    powers = torch.arange(0, 96, 2, device="cuda").float() / 96
    inverse = 1.0 / (factors * 10000.0**powers)
    angles = positions.float()[:, None] * inverse[None, :]
    amplitude = math.sqrt(1 + math.log(32) / math.log(4096))
    cosine, sine = (angles.cos() * amplitude).half()[:, None, :], (angles.sin() * amplitude).half()[:, None, :]
    first, second = x[..., :48], x[..., 48:96]
    expected = torch.cat((first * cosine - second * sine,
                          second * cosine + first * sine, x[..., 96:]), dim=-1).reshape(3, 256)

    context = engine.create_execution_context()
    output = torch.empty(3, 256, device="cuda", dtype=torch.float16)
    for name, tensor in (("x", x), ("position_id", positions),
                         ("rope_long_factor", regime), ("y", output)):
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
