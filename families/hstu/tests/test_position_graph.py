# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the production positional stage against upstream rounding semantics."""

import math

import numpy as np
import pytest

from families.hstu.model import _Graph, load_weights
from families.hstu.tests.fixtures import make_checkpoint


@pytest.mark.gpu
@pytest.mark.parametrize("timestamps", [False, True], ids=["position", "timestamp"])
def test_position_stage_preserves_full_scale_and_materialization(tmp_path, timestamps):
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native positional graph execution requires a CUDA device")

    rows, width = 128, 24
    config = make_checkpoint(
        tmp_path, hidden_size=width, num_layers=1, max_sequence_length=rows,
        position_buckets=16, time_buckets=2048 if timestamps else 0,
        embedding_tables=[{"name": "item", "role": "item", "num_embeddings": rows}],
    )
    weights = load_weights(tmp_path, config)
    weights["embeddings.item.weight"] = np.linspace(-3, 3, rows * width, dtype=np.float32).reshape(rows, width)
    weights["position.weight"] = ((np.arange(16 * width) % 31 - 15) / 32).astype(np.float32).reshape(16, width)
    if timestamps:
        weights["time.weight"] = ((np.arange(2049 * width) % 47 - 23) / 64).astype(np.float32).reshape(2049, width)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(trt, network, config, weights, "bf16")
    graph.outputs()
    # Retain the actual production positional output and let TensorRT prune the
    # downstream blocks. No graph operation or mathematical helper is replaced.
    layers = {network.get_layer(index).name: network.get_layer(index)
              for index in range(network.num_layers)}
    stage = layers.get("position.output", layers["position.add"]).get_output(0)
    assert stage.dtype == trt.bfloat16
    while network.num_outputs:
        network.unmark_output(network.get_output(0))
    stage.name = "position_stage"
    network.mark_output(stage)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = (1, 1, rows, rows) if tensor.name == "attention_weights_transposed" else (1, rows)
        profile.set_shape(tensor.name, shape, shape, shape)
        assert tuple(tuple(bound) for bound in profile.get_shape(tensor.name)) == (shape,) * 3
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    options.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, options)
    assert serialized is not None, "production positional stage must build"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None

    ids = torch.arange(rows, dtype=torch.int32)
    positions, times = ids % 16, (ids * 7) % 2049
    inputs = {"token_ids": ids, "position_ids": positions}
    if timestamps:
        inputs["time_ids"] = times
    inputs = {name: value.reshape(1, rows).cuda() for name, value in inputs.items()}
    inputs["attention_weights_transposed"] = torch.zeros((1, 1, rows, rows), dtype=torch.bfloat16, device="cuda")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = inputs[name]
            assert context.set_input_shape(name, tuple(tensor.shape))
            assert context.set_tensor_address(name, tensor.data_ptr())
    actual = torch.empty((1, rows, width), dtype=torch.bfloat16, device="cuda")
    assert context.set_tensor_address("position_stage", actual.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    raw = torch.from_numpy(weights["embeddings.item.weight"]).bfloat16()
    positional = torch.from_numpy(weights["position.weight"]).bfloat16()[positions.long()]
    if timestamps:
        temporal = torch.from_numpy(weights["time.weight"]).bfloat16()[times.long()]
        # Upstream first uses PyTorch's tensor * Python scalar, materializing a
        # BF16 product, then adds a separately materialized positional sum.
        scaled = raw * math.sqrt(width)
        expected = scaled + (positional + temporal)
        rounded_scale = torch.tensor(math.sqrt(width), dtype=torch.bfloat16)
        assert torch.count_nonzero(scaled != raw * rounded_scale) > rows
    else:
        # The position-only upstream kernel keeps scale and addition in FP32
        # registers until storing the final model-dtype value.
        expected = (raw.float() * math.sqrt(width) + positional.float()).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected.unsqueeze(0), rtol=0, atol=0)
