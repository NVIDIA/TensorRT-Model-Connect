# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One real-TensorRT smoke test for raw graph rewiring."""

import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.tvm_ffi.graph_patch import (  # noqa: E402
    apply_region,
    select_region,
    snapshot_network,
)
from tensorrt_model_connect.trt_compat import network_creation_flags  # noqa: E402


@pytest.mark.gpu
@pytest.mark.trt
def test_rewired_network_compiles() -> None:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(network_creation_flags())
    value = network.add_input("value", trt.float16, (1, 8))
    before = network.add_identity(value)
    selected = network.add_identity(before.get_output(0))
    consumer = network.add_elementwise(
        selected.get_output(0),
        selected.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    network.mark_output(consumer.get_output(0))
    metadata = {"engine_role": "decode"}
    selection = select_region(
        snapshot_network(network, metadata=metadata),
        ["node:1"],
        binding_id="smoke",
    )

    result = apply_region(
        network,
        selection,
        lambda live_network, inputs, _: [
            live_network.add_identity(inputs[0]).get_output(0)
        ],
        metadata=metadata,
    )

    replacement = result.replacement_outputs[0]
    assert consumer.get_input(0) is replacement
    assert consumer.get_input(1) is replacement
    assert builder.build_serialized_network(
        network, builder.create_builder_config()
    ) is not None


@pytest.mark.gpu
@pytest.mark.trt
def test_attention_typed_input_is_captured_and_rewired() -> None:
    if not hasattr(trt.INetworkDefinition, "add_attention_v2"):
        pytest.skip("TensorRT does not expose IAttention")

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(network_creation_flags())
    query = network.add_input("query", trt.float16, (1, 1, 1, 64))
    key = network.add_input("key", trt.float16, (1, 1, 8, 64))
    value = network.add_input("value", trt.float16, (1, 1, 8, 64))
    lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    lengths_layer = network.add_identity(lengths)
    attention = network.add_attention_v2(
        query,
        key,
        value,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.LOWER_RIGHT,
    )
    attention.key_value_lengths = lengths_layer.get_output(0)
    consumer = network.add_identity(attention.get_output(0))
    network.mark_output(consumer.get_output(0))
    metadata = {"engine_role": "decode"}
    attentions = (attention,)
    snapshot = snapshot_network(
        network,
        metadata=metadata,
        attentions=attentions,
    )

    attention_region = select_region(
        snapshot,
        ["node:1", "node:2"],
        binding_id="attention",
    )
    assert len(attention_region.input_tensor_ids) == 4
    assert snapshot.nodes[1].inputs[6] == attention_region.input_tensor_ids[3]

    lengths_region = select_region(snapshot, ["node:0"], binding_id="lengths")
    result = apply_region(
        network,
        lengths_region,
        lambda live_network, inputs, _: [
            live_network.add_identity(inputs[0]).get_output(0)
        ],
        metadata=metadata,
        attentions=attentions,
    )
    assert attention.key_value_lengths is result.replacement_outputs[0]
