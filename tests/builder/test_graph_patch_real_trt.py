# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-TensorRT smoke coverage for multi-instance graph rewiring."""

from __future__ import annotations

from tests.builder.conftest import requires_trt

from tensorrt_model_connect.graph_patch import (
    GraphRegionSelection,
    GraphRegionSelectionSet,
    LayerIdentityContract,
    capture_network,
    compute_region_boundary,
    rewire_selection_set,
)


@requires_trt
def test_multi_instance_rewire_compiles_real_tensorrt_network() -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    identity_contract = LayerIdentityContract(
        provider_id="tests.real_trt_layer_identity",
        schema_version=1,
    )

    # Model builders must record subtype settings at layer construction time:
    # TensorRT's generic ILayer wrapper does not expose every subtype attribute.
    def identity_provider_for(operations_by_name):
        def provide(layer, _index):
            operation = operations_by_name.get(layer.name)
            return {} if operation is None else {"operation": operation}

        return provide

    def elementwise_fingerprint(operation):
        candidate = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        candidate_input = candidate.add_input("value", trt.float32, (1, 4))
        candidate_layer = candidate.add_elementwise(
            candidate_input,
            candidate_input,
            operation,
        )
        candidate_layer.name = "combine"
        candidate_layer.get_output(0).name = "combined"
        candidate.mark_output(candidate_layer.get_output(0))
        identity_provider = identity_provider_for({"combine": str(operation)})
        return capture_network(
            candidate,
            name="real-trt-elementwise",
            metadata={"engine_role": "identity-regression"},
            identity_provider=identity_provider,
            identity_contract=identity_contract,
        ).snapshot.fingerprint

    assert elementwise_fingerprint(trt.ElementWiseOperation.SUM) != elementwise_fingerprint(
        trt.ElementWiseOperation.PROD
    )

    identity_provider = identity_provider_for({})
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    value = network.add_input("value", trt.float32, (1, 4))

    first = network.add_identity(value)
    first.name = "layer.0.replace_me"
    first.get_output(0).name = "first_output"
    network.mark_output(first.get_output(0))
    bridge = network.add_identity(first.get_output(0))
    bridge.name = "layer.0.bridge"
    second = network.add_identity(bridge.get_output(0))
    second.name = "layer.1.replace_me"
    consumer = network.add_identity(second.get_output(0))
    consumer.name = "consumer"
    consumer.get_output(0).name = "output"
    network.mark_output(consumer.get_output(0))

    captured = capture_network(
        network,
        name="real-trt-multi-instance",
        metadata={"engine_role": "decode"},
        identity_provider=identity_provider,
        identity_contract=identity_contract,
    )
    snapshot = captured.snapshot
    assert captured.node_id_for(first) == "node:0"
    assert captured.tensor_id_for(first.get_output(0)) == snapshot.nodes[0].outputs[0]
    selections = []
    for layer_index, node_id in enumerate(("node:0", "node:2")):
        boundary = compute_region_boundary(snapshot, (node_id,))
        selections.append(
            GraphRegionSelection(
                graph_fingerprint=snapshot.fingerprint,
                selected_node_ids=boundary.selected_node_ids,
                input_tensor_ids=boundary.input_tensor_ids,
                output_tensor_ids=boundary.output_tensor_ids,
                stage="decode",
                instance={"layer_index": layer_index},
            )
        )
    selection_set = GraphRegionSelectionSet(
        graph_fingerprint=snapshot.fingerprint,
        selections=tuple(selections),
        stage="decode",
    )

    replacement_outputs = []

    def replacement(live_network, inputs, _artifact):
        layer = live_network.add_identity(inputs[0])
        layer.name = f"external.replacement.{len(replacement_outputs)}"
        output = layer.get_output(0)
        replacement_outputs.append(output)
        return (output,)

    result = rewire_selection_set(
        network,
        snapshot,
        selection_set,
        replacement,
        current_name="real-trt-multi-instance",
        current_metadata={"engine_role": "decode"},
        identity_provider=identity_provider,
        identity_contract=identity_contract,
    )

    assert result.region_count == 2
    assert result.rewired_consumer_inputs == 2
    assert bridge.get_input(0) is replacement_outputs[0]
    assert consumer.get_input(0) is replacement_outputs[1]
    assert [network.get_output(index).name for index in range(network.num_outputs)] == [
        "first_output",
        "output",
    ]

    plan = builder.build_serialized_network(
        network,
        builder.create_builder_config(),
    )
    assert plan is not None
    assert len(bytes(plan)) > 0
