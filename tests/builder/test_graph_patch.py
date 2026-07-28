# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from tensorrt_model_connect.graph_patch import (
    GraphPatchError,
    GraphRegionSelection,
    GraphRegionSelectionSet,
    GraphSnapshot,
    LayerIdentityContract,
    RegionArtifact,
    capture_network,
    coerce_region_selection_set,
    compute_region_boundary,
    create_region_artifact,
    rewire_region,
    rewire_selection,
    rewire_selection_set,
    snapshot_network,
)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...] = (1, 8, 128),
        dtype: str = "float16",
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class EqualFakeTensor(FakeTensor):
    """Distinct backend handles that intentionally compare equal."""

    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EqualFakeTensor)


class ReadOnlyNameFakeTensor:
    def __init__(self, name: str):
        self._name = name
        self.shape = (1, 8, 128)
        self.dtype = "float16"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, _value: str) -> None:
        raise AttributeError("name is read-only")


class FakeLayer:
    def __init__(
        self,
        name: str,
        op: str,
        inputs: list[FakeTensor],
        outputs: list[FakeTensor],
    ):
        self.name = name
        self.type = op
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.set_input_calls: list[tuple[int, FakeTensor]] = []

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def get_input(self, index: int) -> FakeTensor:
        return self.inputs[index]

    def get_output(self, index: int) -> FakeTensor:
        return self.outputs[index]

    def set_input(self, index: int, value: FakeTensor) -> None:
        self.set_input_calls.append((index, value))
        self.inputs[index] = value


class FakeNetwork:
    def __init__(
        self,
        inputs: list[FakeTensor],
        layers: list[FakeLayer],
        outputs: list[FakeTensor],
        name: str = "qwen3_decode",
    ):
        self.name = name
        self.inputs = list(inputs)
        self.layers = list(layers)
        self.outputs = list(outputs)
        self.output_calls: list[tuple[str, FakeTensor]] = []

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    def get_input(self, index: int) -> FakeTensor:
        return self.inputs[index]

    def get_layer(self, index: int) -> FakeLayer:
        return self.layers[index]

    def get_output(self, index: int) -> FakeTensor:
        return self.outputs[index]

    def unmark_output(self, tensor: FakeTensor) -> None:
        self.output_calls.append(("unmark", tensor))
        self.outputs.remove(tensor)

    def mark_output(self, tensor: FakeTensor) -> None:
        self.output_calls.append(("mark", tensor))
        self.outputs.append(tensor)


FAKE_IDENTITY_CONTRACT = LayerIdentityContract(
    provider_id="tests.fake_layer_identity",
    schema_version=1,
)


def _fake_layer_identity(layer: FakeLayer, _index: int) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if hasattr(layer, "operation"):
        attributes["operation"] = str(layer.operation)
    return attributes


def _complete_snapshot(network: FakeNetwork, **kwargs: Any) -> GraphSnapshot:
    return snapshot_network(
        network,
        identity_provider=_fake_layer_identity,
        identity_contract=FAKE_IDENTITY_CONTRACT,
        **kwargs,
    )


def _current_identity(
    network: FakeNetwork,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "current_name": network.name,
        "current_metadata": {} if metadata is None else metadata,
        "identity_provider": _fake_layer_identity,
        "identity_contract": FAKE_IDENTITY_CONTRACT,
    }


def _attention_network() -> tuple[FakeNetwork, dict[str, FakeTensor]]:
    tensors = {
        name: FakeTensor(name) for name in ("hidden", "q", "k", "context", "residual", "tap")
    }
    layers = [
        FakeLayer(
            "model.layers.0.self_attn.q_proj",
            "MATRIX_MULTIPLY",
            [tensors["hidden"]],
            [tensors["q"]],
        ),
        FakeLayer(
            "model.layers.0.self_attn.k_proj",
            "MATRIX_MULTIPLY",
            [tensors["hidden"]],
            [tensors["k"]],
        ),
        FakeLayer(
            "model.layers.0.self_attn.core",
            "SCALED_DOT_PRODUCT_ATTENTION",
            [tensors["q"], tensors["k"]],
            [tensors["context"]],
        ),
        FakeLayer(
            "model.layers.0.residual",
            "ELEMENTWISE",
            [tensors["context"], tensors["hidden"]],
            [tensors["residual"]],
        ),
        FakeLayer(
            "debug.attention_tap",
            "IDENTITY",
            [tensors["context"]],
            [tensors["tap"]],
        ),
    ]
    network = FakeNetwork(
        [tensors["hidden"]],
        layers,
        [tensors["residual"], tensors["context"]],
    )
    return network, tensors


def _elementwise_network(operation: str) -> FakeNetwork:
    value = FakeTensor("value")
    output = FakeTensor("output")
    layer = FakeLayer("combine", "ELEMENTWISE", [value, value], [output])
    layer.operation = operation
    return FakeNetwork([value], [layer], [output], name="elementwise")


def test_snapshot_json_round_trip_and_structural_fingerprint() -> None:
    network, _ = _attention_network()

    def provenance(layer: FakeLayer, index: int) -> dict:
        return {
            "module_path": layer.name.rsplit(".", 1)[0],
            "source": {"file": "graph_blocks.py", "line": 200 + index},
        }

    snapshot = snapshot_network(
        network,
        metadata={"stage": "decode", "model": "Qwen/Qwen3-8B"},
        provenance=provenance,
    )
    restored = GraphSnapshot.from_json(snapshot.to_json())

    assert restored == snapshot
    assert restored.fingerprint == snapshot.fingerprint
    assert restored.identity_contract is None
    assert restored.identity_complete is False
    assert all(node.identity_complete is False for node in restored.nodes)
    assert len(snapshot.fingerprint) == 64
    assert snapshot.nodes[2].provenance == {
        "module_path": "model.layers.0.self_attn",
        "source": {"file": "graph_blocks.py", "line": 202},
    }

    changed = snapshot_network(
        network,
        metadata={"stage": "prefill", "model": "Qwen/Qwen3-8B"},
        provenance=provenance,
    )
    assert changed.fingerprint != snapshot.fingerprint

    moved_source_line = snapshot_network(
        network,
        metadata={"stage": "decode", "model": "Qwen/Qwen3-8B"},
        provenance=lambda layer, index: {
            "module_path": layer.name.rsplit(".", 1)[0],
            "source": {"file": "graph_blocks.py", "line": 900 + index},
        },
    )
    assert moved_source_line.nodes[2].provenance != snapshot.nodes[2].provenance
    assert moved_source_line.fingerprint == snapshot.fingerprint


def test_complete_identity_round_trip_and_operation_drift_changes_fingerprint() -> None:
    summed = _complete_snapshot(_elementwise_network("SUM"))
    multiplied = _complete_snapshot(_elementwise_network("PROD"))
    restored = GraphSnapshot.from_json(summed.to_json())

    assert restored == summed
    assert restored.identity_contract == FAKE_IDENTITY_CONTRACT
    assert restored.identity_complete is True
    assert restored.nodes[0].identity_attributes == {"operation": "SUM"}
    assert summed.fingerprint != multiplied.fingerprint


def test_complete_identity_serialization_rejects_tampering() -> None:
    snapshot = _complete_snapshot(_elementwise_network("SUM"))

    changed_attributes = snapshot.to_dict()
    changed_attributes["nodes"][0]["identity"]["attributes"]["operation"] = "PROD"
    with pytest.raises(GraphPatchError, match="fingerprint mismatch"):
        GraphSnapshot.from_dict(changed_attributes)

    changed_completeness = snapshot.to_dict()
    changed_completeness["identity"]["complete"] = False
    with pytest.raises(GraphPatchError, match="completeness does not match"):
        GraphSnapshot.from_dict(changed_completeness)


def test_identity_provider_requires_contract_and_canonical_json() -> None:
    network, _ = _attention_network()

    with pytest.raises(GraphPatchError, match="both be present"):
        snapshot_network(network, identity_provider=_fake_layer_identity)
    with pytest.raises(GraphPatchError, match="both be present"):
        snapshot_network(network, identity_contract=FAKE_IDENTITY_CONTRACT)
    with pytest.raises(GraphPatchError, match="contract has the wrong type"):
        snapshot_network(
            network,
            identity_provider=_fake_layer_identity,
            identity_contract={  # type: ignore[arg-type]
                "provider_id": "tests.fake_layer_identity",
                "schema_version": 1,
            },
        )
    with pytest.raises(GraphPatchError, match="JSON scalars"):
        snapshot_network(
            network,
            identity_provider=lambda _layer, _index: {"unstable": {"q", "k"}},
            identity_contract=FAKE_IDENTITY_CONTRACT,
        )

    partial = snapshot_network(
        network,
        identity_provider=lambda _layer, index: None if index == 2 else {},
        identity_contract=FAKE_IDENTITY_CONTRACT,
    )
    assert partial.identity_complete is False
    assert partial.nodes[2].identity_complete is False


def test_snapshot_rejects_non_json_metadata_values() -> None:
    network, _ = _attention_network()

    with pytest.raises(GraphPatchError, match="JSON scalars"):
        snapshot_network(network, metadata={"unstable": {"q", "k"}})


def test_snapshot_rejects_non_string_metadata_keys_before_normalization() -> None:
    network, _ = _attention_network()

    with pytest.raises(GraphPatchError, match="keys must be strings"):
        snapshot_network(
            network,
            metadata={"collision": {1: "integer", "1": "string"}},
        )


def test_snapshot_tracks_tensor_handles_by_identity_not_equality() -> None:
    graph_input = EqualFakeTensor("input")
    first_output = EqualFakeTensor("first")
    second_output = EqualFakeTensor("second")
    network = FakeNetwork(
        [graph_input],
        [
            FakeLayer("first", "IDENTITY", [graph_input], [first_output]),
            FakeLayer("second", "IDENTITY", [first_output], [second_output]),
        ],
        [second_output],
    )

    captured = capture_network(network)
    snapshot = captured.snapshot

    assert len(snapshot.tensors) == 3
    assert snapshot.nodes[0].outputs == ("tensor:1",)
    assert snapshot.nodes[1].inputs == ("tensor:1",)
    assert snapshot.nodes[1].outputs == ("tensor:2",)
    assert captured.node_id_for(network.layers[0]) == "node:0"
    equivalent_wrapper = FakeLayer(
        "first",
        "IDENTITY",
        [graph_input],
        [first_output],
    )
    assert captured.node_id_for(equivalent_wrapper) == "node:0"
    assert captured.tensor_id_for(first_output) == "tensor:1"
    with pytest.raises(GraphPatchError, match="does not belong"):
        captured.tensor_id_for(EqualFakeTensor("unknown"))


def test_captured_graph_resolves_unique_no_output_layer_proxy() -> None:
    graph_input = FakeTensor("input")
    network = FakeNetwork(
        [graph_input],
        [FakeLayer("sink", "ASSERTION", [graph_input], [])],
        [graph_input],
    )

    captured = capture_network(network)
    equivalent_wrapper = FakeLayer("sink", "ASSERTION", [graph_input], [])

    assert captured.node_id_for(equivalent_wrapper) == "node:0"


def test_captured_graph_rejects_ambiguous_no_output_layer_proxy() -> None:
    graph_input = FakeTensor("input")
    network = FakeNetwork(
        [graph_input],
        [
            FakeLayer("sink", "ASSERTION", [graph_input], []),
            FakeLayer("sink", "ASSERTION", [graph_input], []),
        ],
        [graph_input],
    )

    captured = capture_network(network)
    equivalent_wrapper = FakeLayer("sink", "ASSERTION", [graph_input], [])

    with pytest.raises(GraphPatchError, match="does not belong"):
        captured.node_id_for(equivalent_wrapper)


def test_captured_graph_rejects_one_layer_handle_used_for_two_nodes() -> None:
    graph_input = FakeTensor("input")
    shared_layer = FakeLayer("sink", "ASSERTION", [graph_input], [])
    network = FakeNetwork(
        [graph_input],
        [shared_layer, shared_layer],
        [graph_input],
    )

    captured = capture_network(network)

    with pytest.raises(GraphPatchError, match="does not belong"):
        captured.node_id_for(shared_layer)


def test_boundary_and_region_artifact_are_derived_from_selection() -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)

    boundary = compute_region_boundary(snapshot, ["node:2"])

    assert boundary.selected_node_ids == ("node:2",)
    assert boundary.input_tensor_ids == ("tensor:1", "tensor:2")
    assert boundary.output_tensor_ids == ("tensor:3",)
    assert boundary.internal_tensor_ids == ()

    artifact = create_region_artifact(
        snapshot,
        ["node:2"],
        metadata={"requested_replacement": "tvm_ffi"},
    )
    restored = RegionArtifact.from_json(artifact.to_json())
    assert restored == artifact
    assert restored.graph_fingerprint == snapshot.fingerprint
    assert restored.input_tensor_ids == ("tensor:1", "tensor:2")
    assert restored.output_tensor_ids == ("tensor:3",)
    assert {tensor.name for tensor in restored.tensors} == {"q", "k", "context"}
    assert len(restored.fingerprint) == 64


def test_multi_node_region_records_internal_tensor() -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)

    boundary = compute_region_boundary(snapshot, ["node:0", "node:2"])

    assert boundary.selected_node_ids == ("node:0", "node:2")
    assert boundary.input_tensor_ids == ("tensor:0", "tensor:2")
    assert boundary.output_tensor_ids == ("tensor:3",)
    assert boundary.internal_tensor_ids == ("tensor:1",)


def test_rewire_region_updates_all_external_consumers_and_network_output() -> None:
    network, tensors = _attention_network()

    def provenance(layer: FakeLayer, _index: int) -> dict:
        return {"module_path": layer.name}

    snapshot = _complete_snapshot(network, provenance=provenance)
    new_context = FakeTensor("ffi_context")
    callback_receipt: dict[str, object] = {}

    def replacement(
        received_network: FakeNetwork,
        inputs: tuple[FakeTensor, ...],
        artifact: RegionArtifact,
    ) -> list[FakeTensor]:
        callback_receipt["network"] = received_network
        callback_receipt["inputs"] = inputs
        callback_receipt["artifact"] = artifact
        return [new_context]

    result = rewire_region(
        network,
        snapshot,
        ["node:2"],
        replacement,
        provenance=provenance,
        **_current_identity(network),
    )

    assert callback_receipt["network"] is network
    assert callback_receipt["inputs"] == (tensors["q"], tensors["k"])
    assert callback_receipt["artifact"].selected_node_ids == ("node:2",)
    assert network.layers[3].inputs[0] is new_context
    assert network.layers[4].inputs[0] is new_context
    assert network.layers[2].outputs[0] is tensors["context"]
    assert result.rewired_consumer_inputs == 2
    assert result.rewired_network_outputs == 1
    assert result.replacement_outputs == (new_context,)
    assert network.output_calls == [
        ("unmark", tensors["residual"]),
        ("unmark", tensors["context"]),
        ("mark", tensors["residual"]),
        ("mark", new_context),
    ]
    assert new_context.name == "context"
    assert network.outputs == [tensors["residual"], new_context]


def test_rewire_region_preserves_network_output_order() -> None:
    network, tensors = _attention_network()
    network.outputs = [tensors["context"], tensors["residual"]]
    snapshot = _complete_snapshot(network)
    new_context = FakeTensor("ffi_context")

    rewire_region(
        network,
        snapshot,
        ["node:2"],
        lambda *_args: [new_context],
        **_current_identity(network),
    )

    assert network.outputs == [new_context, tensors["residual"]]
    assert new_context.name == "context"


def test_rewire_region_fails_before_mutation_when_output_name_cannot_be_preserved() -> None:
    network, tensors = _attention_network()
    snapshot = _complete_snapshot(network)
    replacement = ReadOnlyNameFakeTensor("ffi_context")

    with pytest.raises(GraphPatchError, match="preserve the public name"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            lambda *_args: [replacement],
            **_current_identity(network),
        )

    assert network.layers[3].inputs[0] is tensors["context"]
    assert network.layers[4].inputs[0] is tensors["context"]
    assert network.outputs == [tensors["residual"], tensors["context"]]
    assert network.output_calls == []
    assert tensors["context"].name == "context"


@pytest.mark.parametrize(
    ("field", "expected", "actual"),
    [
        ("dtype", "float32", "int8"),
        ("shape", (1,), (99,)),
        ("location", "DEVICE", "HOST"),
        ("is_shape_tensor", False, True),
    ],
)
def test_rewire_region_rejects_output_abi_mismatch_before_mutation(
    field: str,
    expected: object,
    actual: object,
) -> None:
    network, tensors = _attention_network()
    setattr(tensors["context"], field, expected)
    snapshot = _complete_snapshot(network)
    replacement = FakeTensor("ffi_context")
    setattr(replacement, field, actual)

    with pytest.raises(GraphPatchError, match=rf"ABI mismatch.*{field}"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            lambda *_args: [replacement],
            **_current_identity(network),
        )

    assert network.layers[3].inputs[0] is tensors["context"]
    assert network.layers[4].inputs[0] is tensors["context"]
    assert network.outputs == [tensors["residual"], tensors["context"]]
    assert network.output_calls == []


@pytest.mark.parametrize("tensor_name", ["hidden", "residual", "context"])
def test_rewire_region_rejects_original_tensor_as_replacement_output(
    tensor_name: str,
) -> None:
    network, tensors = _attention_network()
    snapshot = _complete_snapshot(network)
    original_names = {name: tensor.name for name, tensor in tensors.items()}

    with pytest.raises(GraphPatchError, match="newly created backend tensors"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            lambda *_args: [tensors[tensor_name]],
            **_current_identity(network),
        )

    assert network.layers[3].inputs[0] is tensors["context"]
    assert network.layers[4].inputs[0] is tensors["context"]
    assert network.outputs == [tensors["residual"], tensors["context"]]
    assert network.output_calls == []
    assert {name: tensor.name for name, tensor in tensors.items()} == original_names


def test_rewire_fails_closed_when_graph_drifted() -> None:
    network, _ = _attention_network()
    snapshot = _complete_snapshot(network)
    network.layers[2].name = "model.layers.0.self_attn.changed"
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return []

    with pytest.raises(GraphPatchError, match="no longer matches"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            replacement,
            **_current_identity(network),
        )

    assert called is False


@pytest.mark.parametrize("entrypoint", ["region", "selection", "selection_set"])
def test_all_rewire_entrypoints_reject_structural_only_snapshot(
    entrypoint: str,
) -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    boundary = compute_region_boundary(snapshot, ["node:2"])
    selection = GraphRegionSelection(
        graph_fingerprint=snapshot.fingerprint,
        selected_node_ids=boundary.selected_node_ids,
        input_tensor_ids=boundary.input_tensor_ids,
        output_tensor_ids=boundary.output_tensor_ids,
    )
    selection_set = GraphRegionSelectionSet(
        graph_fingerprint=snapshot.fingerprint,
        selections=(selection,),
    )
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="structural-only or incomplete"):
        if entrypoint == "region":
            rewire_region(
                network,
                snapshot,
                ["node:2"],
                replacement,
                **_current_identity(network),
            )
        elif entrypoint == "selection":
            rewire_selection(
                network,
                snapshot,
                selection,
                replacement,
                **_current_identity(network),
            )
        else:
            rewire_selection_set(
                network,
                snapshot,
                selection_set,
                replacement,
                **_current_identity(network),
            )

    assert called is False
    assert network.output_calls == []


def test_rewire_rejects_layer_identity_operation_drift_before_callback() -> None:
    network = _elementwise_network("SUM")
    snapshot = _complete_snapshot(network)
    network.layers[0].operation = "PROD"
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="no longer matches"):
        rewire_region(
            network,
            snapshot,
            ["node:0"],
            replacement,
            **_current_identity(network),
        )

    assert called is False


@pytest.mark.parametrize(
    ("current_name", "current_metadata"),
    [
        ("other-graph", {"precision": "fp16"}),
        ("qwen3_decode", {"precision": "bf16"}),
    ],
)
def test_rewire_validates_current_name_and_metadata_instead_of_replaying_snapshot(
    current_name: str,
    current_metadata: dict[str, str],
) -> None:
    network, _ = _attention_network()
    expected_metadata = {"precision": "fp16"}
    snapshot = _complete_snapshot(network, metadata=expected_metadata)
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="no longer matches"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            replacement,
            current_name=current_name,
            current_metadata=current_metadata,
            identity_provider=_fake_layer_identity,
            identity_contract=FAKE_IDENTITY_CONTRACT,
        )

    assert called is False


def test_rewire_rejects_contract_mismatch_or_incomplete_current_provider() -> None:
    network, _ = _attention_network()
    snapshot = _complete_snapshot(network)
    selection = _selection_for_nodes(snapshot, ["node:2"])
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="contract does not match"):
        rewire_selection(
            network,
            snapshot,
            selection,
            replacement,
            current_name=network.name,
            current_metadata={},
            identity_provider=_fake_layer_identity,
            identity_contract=LayerIdentityContract(
                provider_id="tests.other_identity",
                schema_version=1,
            ),
        )
    with pytest.raises(GraphPatchError, match="did not completely describe"):
        rewire_selection(
            network,
            snapshot,
            selection,
            replacement,
            current_name=network.name,
            current_metadata={},
            identity_provider=lambda _layer, index: None if index == 2 else {},
            identity_contract=FAKE_IDENTITY_CONTRACT,
        )

    assert called is False


def test_invalid_selection_and_output_count_are_rejected() -> None:
    network, _ = _attention_network()
    snapshot = _complete_snapshot(network)

    with pytest.raises(GraphPatchError, match="at least one"):
        compute_region_boundary(snapshot, [])
    with pytest.raises(GraphPatchError, match="unknown node"):
        compute_region_boundary(snapshot, ["node:99"])
    with pytest.raises(GraphPatchError, match="returned 0 outputs"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            lambda *_args: [],
            **_current_identity(network),
        )


def _selection_document(snapshot: GraphSnapshot) -> dict:
    return {
        "schema_version": 1,
        "kind": "tensorrt_model_connect.graph_region",
        "graph": {
            "model": "Qwen/Qwen3-8B",
            "stage": "decode",
            "fingerprint": snapshot.fingerprint,
        },
        "selection": {
            "node_ids": ["node:2"],
        },
        "boundary": {
            "inputs": [
                {"tensor_id": "tensor:1"},
                {"tensor_id": "tensor:2"},
            ],
            "outputs": [
                {"tensor_id": "tensor:3"},
            ],
        },
    }


def _selection_for_nodes(
    snapshot: GraphSnapshot,
    node_ids: list[str],
) -> GraphRegionSelection:
    boundary = compute_region_boundary(snapshot, node_ids)
    return GraphRegionSelection(
        graph_fingerprint=snapshot.fingerprint,
        selected_node_ids=boundary.selected_node_ids,
        input_tensor_ids=boundary.input_tensor_ids,
        output_tensor_ids=boundary.output_tensor_ids,
    )


def _repeated_regions_network() -> tuple[FakeNetwork, dict[str, FakeTensor]]:
    tensors = {
        name: FakeTensor(name)
        for name in (
            "hidden",
            "context_0",
            "hidden_1",
            "context_1",
            "output",
        )
    }
    network = FakeNetwork(
        [tensors["hidden"]],
        [
            FakeLayer(
                "model.layers.0.self_attn.core",
                "SCALED_DOT_PRODUCT_ATTENTION",
                [tensors["hidden"]],
                [tensors["context_0"]],
            ),
            FakeLayer(
                "model.layers.0.residual",
                "ELEMENTWISE",
                [tensors["context_0"], tensors["hidden"]],
                [tensors["hidden_1"]],
            ),
            FakeLayer(
                "model.layers.1.self_attn.core",
                "SCALED_DOT_PRODUCT_ATTENTION",
                [tensors["hidden_1"]],
                [tensors["context_1"]],
            ),
            FakeLayer(
                "model.layers.1.residual",
                "ELEMENTWISE",
                [tensors["context_1"], tensors["hidden_1"]],
                [tensors["output"]],
            ),
        ],
        [tensors["output"]],
    )
    return network, tensors


def _selection_set_document(snapshot: GraphSnapshot) -> dict:
    region_0 = compute_region_boundary(snapshot, ["node:0"])
    region_1 = compute_region_boundary(snapshot, ["node:2"])

    def region(boundary, layer_index):
        return {
            "instance": {
                "layer_index": layer_index,
                "layer_prefix": f"model.layers.{layer_index}.self_attn",
            },
            "selection": {"node_ids": list(boundary.selected_node_ids)},
            "boundary": {
                "inputs": [{"tensor_id": tensor_id} for tensor_id in boundary.input_tensor_ids],
                "outputs": [{"tensor_id": tensor_id} for tensor_id in boundary.output_tensor_ids],
            },
        }

    return {
        "schema_version": 1,
        "kind": "tensorrt_model_connect.graph_region_set",
        "graph": {
            "model": "Qwen/Qwen3-8B",
            "stage": "decode",
            "fingerprint": snapshot.fingerprint,
            "build": {"precision": "fp16"},
        },
        "regions": [
            region(region_0, 0),
            region(region_1, 1),
        ],
    }


def test_complete_selection_document_drives_exact_rewire() -> None:
    network, tensors = _attention_network()
    snapshot = _complete_snapshot(network)
    selection = GraphRegionSelection.from_dict(_selection_document(snapshot))
    replacement_output = FakeTensor("ffi_context")

    result = rewire_selection(
        network,
        snapshot,
        selection,
        lambda _network, inputs, _artifact: (
            [replacement_output] if inputs == (tensors["q"], tensors["k"]) else []
        ),
        **_current_identity(network),
    )

    assert selection.input_tensor_ids == ("tensor:1", "tensor:2")
    assert selection.output_tensor_ids == ("tensor:3",)
    assert result.rewired_consumer_inputs == 2
    assert network.layers[3].inputs[0] is replacement_output
    assert network.layers[4].inputs[0] is replacement_output


def test_region_set_parses_single_region_and_round_trips() -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    selection = GraphRegionSelection.from_dict(_selection_document(snapshot))

    wrapped = GraphRegionSelectionSet.from_dict(_selection_document(snapshot))
    assert wrapped == coerce_region_selection_set(selection)
    assert wrapped.selections == (selection,)
    assert GraphRegionSelection.from_json(selection.to_json()) == selection

    repeated_network, _ = _repeated_regions_network()
    repeated_snapshot = snapshot_network(repeated_network)
    selection_set = GraphRegionSelectionSet.from_dict(_selection_set_document(repeated_snapshot))
    restored = GraphRegionSelectionSet.from_json(selection_set.to_json())

    assert restored == selection_set
    assert len(restored.selections) == 2
    assert restored.selections[1].instance["layer_index"] == 1
    assert restored.build == {"precision": "fp16"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["selection"].update({"semantic_paths": []}),
        lambda value: value["boundary"]["outputs"][0].update({"consumer": "node:3"}),
        lambda value: value["boundary"]["outputs"].append({"tensor_id": "tensor:3"}),
    ],
)
def test_selection_document_rejects_unknown_or_duplicate_boundary_data(
    mutation,
) -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    value = _selection_document(snapshot)
    mutation(value)

    with pytest.raises(GraphPatchError, match="Unknown|duplicate"):
        GraphRegionSelection.from_dict(value)


@pytest.mark.parametrize(
    "field",
    ["node_id", "model", "stage"],
)
def test_selection_document_rejects_non_string_fields(field: str) -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    value = _selection_document(snapshot)
    if field == "node_id":
        value["selection"]["node_ids"] = [2]
    else:
        value["graph"][field] = {"not": "a string"}

    with pytest.raises(GraphPatchError, match="string|node ID"):
        GraphRegionSelection.from_dict(value)


@pytest.mark.parametrize("field", ["kind", "schema_version"])
def test_selection_document_requires_kind_and_schema_version(field: str) -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    value = _selection_document(snapshot)
    value.pop(field)

    with pytest.raises(GraphPatchError, match="Unsupported"):
        GraphRegionSelection.from_dict(value)


def test_region_set_replaces_each_independent_instance_once() -> None:
    network, tensors = _repeated_regions_network()
    snapshot = _complete_snapshot(network)
    selection_set = GraphRegionSelectionSet.from_dict(_selection_set_document(snapshot))
    replacements = [
        FakeTensor("ffi_context_0"),
        FakeTensor("ffi_context_1"),
    ]
    calls = []

    def replacement(received_network, inputs, artifact):
        calls.append((received_network, inputs, artifact))
        return [replacements[len(calls) - 1]]

    result = rewire_selection_set(
        network,
        snapshot,
        selection_set,
        replacement,
        **_current_identity(network),
    )

    assert len(calls) == 2
    assert calls[0][1] == (tensors["hidden"],)
    assert calls[1][1] == (tensors["hidden_1"],)
    assert calls[0][2].metadata["instance"]["layer_index"] == 0
    assert calls[1][2].metadata["instance"]["layer_index"] == 1
    assert network.layers[1].inputs[0] is replacements[0]
    assert network.layers[3].inputs[0] is replacements[1]
    assert result.region_count == 2
    assert result.selected_node_count == 2
    assert result.rewired_consumer_inputs == 2
    assert result.rewired_network_outputs == 0
    assert result.artifacts == tuple(call[2] for call in calls)
    with pytest.raises(GraphPatchError, match="no single artifact"):
        _ = result.artifact


def test_region_set_rejects_overlap_and_direct_dependencies() -> None:
    network, _ = _repeated_regions_network()
    snapshot = snapshot_network(network)
    value = _selection_set_document(snapshot)
    value["regions"][1] = value["regions"][0]
    with pytest.raises(GraphPatchError, match="must not overlap"):
        GraphRegionSelectionSet.from_dict(value)

    first = _selection_for_nodes(snapshot, ["node:0"])
    directly_dependent = _selection_for_nodes(snapshot, ["node:1"])
    selection_set = GraphRegionSelectionSet(
        graph_fingerprint=snapshot.fingerprint,
        selections=(first, directly_dependent),
    )
    with pytest.raises(GraphPatchError, match="directly depend"):
        selection_set.validate(snapshot)


def test_region_set_validates_all_outputs_before_rewiring() -> None:
    network, tensors = _repeated_regions_network()
    snapshot = _complete_snapshot(network)
    selection_set = GraphRegionSelectionSet.from_dict(_selection_set_document(snapshot))
    calls = 0

    def replacement(_network, _inputs, _artifact):
        nonlocal calls
        calls += 1
        return [FakeTensor("first")] if calls == 1 else []

    with pytest.raises(GraphPatchError, match="region 1"):
        rewire_selection_set(
            network,
            snapshot,
            selection_set,
            replacement,
            **_current_identity(network),
        )

    assert calls == 2
    assert network.layers[1].inputs[0] is tensors["context_0"]
    assert network.layers[3].inputs[0] is tensors["context_1"]


def test_region_set_rejects_one_tensor_for_two_region_outputs_before_rewiring() -> None:
    network, tensors = _repeated_regions_network()
    snapshot = _complete_snapshot(network)
    selection_set = GraphRegionSelectionSet.from_dict(_selection_set_document(snapshot))
    shared = FakeTensor("shared")
    original_output_names = [tensor.name for tensor in network.outputs]

    with pytest.raises(GraphPatchError, match="two region outputs"):
        rewire_selection_set(
            network,
            snapshot,
            selection_set,
            lambda *_args: [shared],
            **_current_identity(network),
        )

    assert network.layers[1].inputs[0] is tensors["context_0"]
    assert network.layers[3].inputs[0] is tensors["context_1"]
    assert network.outputs == [tensors["output"]]
    assert [tensor.name for tensor in network.outputs] == original_output_names
    assert shared.name == "shared"


def test_selection_document_fails_closed_on_fingerprint_or_boundary_drift() -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    value = _selection_document(snapshot)
    value["graph"]["fingerprint"] = "stale"
    with pytest.raises(GraphPatchError, match="fingerprint"):
        GraphRegionSelection.from_dict(value).validate(snapshot)

    value = _selection_document(snapshot)
    value["boundary"]["outputs"] = [{"tensor_id": "tensor:4"}]
    with pytest.raises(GraphPatchError, match="output boundary"):
        GraphRegionSelection.from_dict(value).validate(snapshot)


def test_selection_rejects_disconnected_nonconvex_and_plugin_regions() -> None:
    network, _ = _attention_network()
    snapshot = snapshot_network(network)
    with pytest.raises(GraphPatchError, match="connected region"):
        _selection_for_nodes(snapshot, ["node:0", "node:1"]).validate(snapshot)

    hidden = FakeTensor("hidden")
    direct = FakeTensor("direct")
    detour = FakeTensor("detour")
    output = FakeTensor("output")
    nonconvex = FakeNetwork(
        [hidden],
        [
            FakeLayer("a", "IDENTITY", [hidden], [direct]),
            FakeLayer("b", "IDENTITY", [direct], [detour]),
            FakeLayer("c", "ELEMENTWISE", [direct, detour], [output]),
        ],
        [output],
    )
    nonconvex_snapshot = _complete_snapshot(nonconvex)
    with pytest.raises(GraphPatchError, match="convex region"):
        _selection_for_nodes(
            nonconvex_snapshot,
            ["node:0", "node:2"],
        ).validate(nonconvex_snapshot)
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="convex region"):
        rewire_region(
            nonconvex,
            nonconvex_snapshot,
            ["node:0", "node:2"],
            replacement,
            **_current_identity(nonconvex),
        )
    assert called is False

    network.layers[2].type = "PLUGIN_V3"
    plugin_snapshot = _complete_snapshot(network)
    with pytest.raises(GraphPatchError, match="existing plugin"):
        _selection_for_nodes(plugin_snapshot, ["node:2"]).validate(plugin_snapshot)
    with pytest.raises(GraphPatchError, match="existing plugin"):
        rewire_region(
            network,
            plugin_snapshot,
            ["node:2"],
            replacement,
            **_current_identity(network),
        )
    assert called is False


@pytest.mark.parametrize(
    "unsafe_op",
    [
        "CONDITIONAL_INPUT",
        "ITERATOR",
        "LOOP_OUTPUT",
        "RECURRENCE",
        "TRIP_LIMIT",
    ],
)
def test_rewire_region_rejects_control_flow_layers(unsafe_op: str) -> None:
    network, _ = _attention_network()
    network.layers[2].type = unsafe_op
    snapshot = _complete_snapshot(network)
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return [FakeTensor("replacement")]

    with pytest.raises(GraphPatchError, match="control-flow"):
        create_region_artifact(snapshot, ["node:2"])
    with pytest.raises(GraphPatchError, match="control-flow"):
        rewire_region(
            network,
            snapshot,
            ["node:2"],
            replacement,
            **_current_identity(network),
        )
    assert called is False


def test_graph_patch_module_has_no_tensorrt_import() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "python" / "tensorrt_model_connect" / "graph_patch.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "tensorrt" not in imported_modules
    assert "tensorrt_rtx" not in imported_modules
