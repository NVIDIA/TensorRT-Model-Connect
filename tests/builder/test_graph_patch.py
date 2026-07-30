# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.tvm_ffi.graph_patch import (
    GraphPatchError,
    apply_region,
    load_selection,
    load_snapshot,
    select_region,
    snapshot_network,
    write_selection,
    write_snapshot,
)
from tensorrt_model_connect.tvm_ffi import graph_build


class FakeTensor:
    def __init__(
        self,
        name: str,
        *,
        dtype: str = "float16",
        shape: tuple[int, ...] = (1, 8),
        is_shape_tensor: bool = False,
        location: str = "DEVICE",
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.is_shape_tensor = is_shape_tensor
        self.location = location


class FakeLayer:
    def __init__(
        self,
        name: str,
        inputs: list[FakeTensor],
        *,
        op: str = "IDENTITY",
        outputs: list[FakeTensor] | None = None,
    ) -> None:
        self.name = name
        self.type = op
        self.inputs = list(inputs)
        self.outputs = outputs or [FakeTensor(name + ".output")]
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

    def set_input(self, index: int, tensor: FakeTensor) -> None:
        self.inputs[index] = tensor
        self.set_input_calls.append((index, tensor))


class FakeNetwork:
    def __init__(
        self,
        inputs: list[FakeTensor],
        layers: list[FakeLayer],
        outputs: list[FakeTensor],
    ) -> None:
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs

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


def _network() -> tuple[FakeNetwork, dict[str, FakeLayer]]:
    value = FakeTensor("value")
    first = FakeLayer("first", [value])
    selected = FakeLayer("replace_me", [first.get_output(0)], op="ACTIVATION")
    left = FakeLayer("left_consumer", [selected.get_output(0)])
    right = FakeLayer("right_consumer", [selected.get_output(0)])
    merge = FakeLayer("merge", [left.get_output(0), right.get_output(0)], op="SUM")
    network = FakeNetwork([value], [first, selected, left, right, merge], [merge.get_output(0)])
    return network, {layer.name: layer for layer in network.layers}


def _snapshot(network: FakeNetwork):
    return snapshot_network(network, metadata={"engine_role": "decode", "precision": "fp16"})


def test_snapshot_and_explicit_selection_round_trip(tmp_path) -> None:
    network, _ = _network()
    snapshot = _snapshot(network)
    assert [node.name for node in snapshot.nodes] == [
        "first",
        "replace_me",
        "left_consumer",
        "right_consumer",
        "merge",
    ]
    assert len(snapshot.fingerprint) == 64

    snapshot_path = tmp_path / "graph.json"
    write_snapshot(snapshot, snapshot_path)
    assert load_snapshot(snapshot_path) == snapshot

    selection = select_region(
        snapshot,
        ["node:1"],
        binding_id="attention",
        workspace_bytes=4096,
        extra_args=[{"type": "float", "value": 0.5}],
    )
    assert selection.engine_role == "decode"
    assert selection.kernel_name == "trtmc.slot.attention"
    assert len(selection.abi_sha256) == 64
    assert selection.node_ids == ("node:1",)
    assert len(selection.input_tensor_ids) == 1
    assert len(selection.output_tensor_ids) == 1

    selection_path = tmp_path / "region.json"
    write_selection(selection, selection_path)
    assert load_selection(selection_path) == selection
    assert set(json.loads(selection_path.read_text())) == {
        "schema_version",
        "graph_fingerprint",
        "engine_role",
        "binding_id",
        "abi_sha256",
        "workspace_bytes",
        "extra_args",
        "node_ids",
        "input_tensor_ids",
        "output_tensor_ids",
        "output_shape_input",
    }


def test_snapshot_round_trip_preserves_duplicate_input_slots(tmp_path) -> None:
    value = FakeTensor("value")
    duplicate = FakeLayer("duplicate", [value, value], op="SUM")
    consumer = FakeLayer("consumer", [duplicate.get_output(0)])
    snapshot = _snapshot(FakeNetwork([value], [duplicate, consumer], [consumer.get_output(0)]))
    path = tmp_path / "graph.json"

    write_snapshot(snapshot, path)

    assert load_snapshot(path) == snapshot
    assert snapshot.tensors[0].consumers == ("node:0", "node:0")


def test_apply_rewires_every_external_consumer() -> None:
    network, layers = _network()
    selection = select_region(_snapshot(network), ["node:1"], binding_id="attention")
    replacement_output = FakeTensor("external.output")
    seen_inputs: tuple[FakeTensor, ...] = ()

    def replacement(_network, inputs, received_selection):
        nonlocal seen_inputs
        seen_inputs = inputs
        assert received_selection is selection
        return [replacement_output]

    result = apply_region(
        network,
        selection,
        replacement,
        metadata={"engine_role": "decode", "precision": "fp16"},
    )
    assert result.rewired_consumer_inputs == 2
    assert seen_inputs == (layers["first"].get_output(0),)
    assert layers["left_consumer"].get_input(0) is replacement_output
    assert layers["right_consumer"].get_input(0) is replacement_output
    assert layers["merge"].set_input_calls == []


def test_build_session_captures_then_applies_one_runtime_slot(monkeypatch, tmp_path) -> None:
    metadata = {"precision": "fp16"}
    snapshot_path = tmp_path / "graph.json"
    network, _ = _network()
    with pytest.raises(graph_build.GraphInspectionComplete):
        with graph_build.inspect_graph(snapshot_path, engine_role="decode", metadata=metadata):
            with graph_build.engine_role("decode"):
                graph_build.process_network(network)

    snapshot = load_snapshot(snapshot_path)
    selection = select_region(snapshot, ["node:1"], binding_id="attention")
    selection_path = tmp_path / "selection.json"
    write_selection(selection, selection_path)

    replacement_output = FakeTensor("ffi.output")
    calls = []

    def add_kernel(live_network, **kwargs):
        calls.append((live_network, kwargs))
        return [replacement_output]

    monkeypatch.setattr(
        "tensorrt_model_connect.tvm_ffi.plugin.add_tvm_ffi_kernel",
        add_kernel,
    )
    network, layers = _network()
    with graph_build.apply_graph_slot(selection_path, metadata=metadata):
        with graph_build.engine_role("decode"):
            graph_build.process_network(network)
        descriptor = json.loads(graph_build.kernel_slots_section())

    assert layers["left_consumer"].get_input(0) is replacement_output
    assert calls[0][1]["kernel_name"] == "trtmc.slot.attention"
    assert descriptor == {
        "schema_version": 1,
        "slots": [
            {
                "abi_sha256": selection.abi_sha256,
                "id": "attention",
            }
        ],
    }


def test_family_recipe_records_an_exact_layer_interval(tmp_path) -> None:
    value = FakeTensor("value")
    network = FakeNetwork([value], [], [])
    snapshot_path = tmp_path / "graph.json"

    with pytest.raises(graph_build.GraphInspectionComplete):
        with graph_build.inspect_graph(
            snapshot_path,
            engine_role="decode",
            metadata={"precision": "fp16"},
        ):
            with graph_build.engine_role("decode"):
                with graph_build.graph_recipe_region(
                    network,
                    "family.activation@1",
                    "decoder.layer.0",
                ):
                    selected = FakeLayer("selected", [value], op="ACTIVATION")
                    network.layers.append(selected)
                consumer = FakeLayer("consumer", [selected.get_output(0)])
                network.layers.append(consumer)
                network.outputs.append(consumer.get_output(0))
                graph_build.process_network(network)

    snapshot = load_snapshot(snapshot_path)
    assert snapshot.metadata["graph_recipes"] == [
        {
            "id": "family.activation@1",
            "instance": "decoder.layer.0",
            "node_ids": ["node:0"],
            "workspace_bytes": 0,
            "extra_args": [],
            "output_shape_input": None,
        }
    ]


def test_apply_rejects_stale_graph_before_callback() -> None:
    network, layers = _network()
    selection = select_region(_snapshot(network), ["node:1"], binding_id="attention")
    layers["first"].name = "changed"
    called = False

    def replacement(*_args):
        nonlocal called
        called = True
        return []

    with pytest.raises(GraphPatchError, match="fingerprint"):
        apply_region(
            network,
            selection,
            replacement,
            metadata={"engine_role": "decode", "precision": "fp16"},
        )
    assert not called


def test_selection_requires_connected_and_convex_region() -> None:
    network, _ = _network()
    snapshot = _snapshot(network)
    with pytest.raises(GraphPatchError, match="connected"):
        select_region(snapshot, ["node:0", "node:4"], binding_id="bad")

    value = FakeTensor("value")
    first = FakeLayer("first", [value])
    outside = FakeLayer("outside", [first.get_output(0)])
    join = FakeLayer("join", [first.get_output(0), outside.get_output(0)])
    consumer = FakeLayer("consumer", [join.get_output(0)])
    nonconvex = FakeNetwork(
        [value],
        [first, outside, join, consumer],
        [consumer.get_output(0)],
    )
    with pytest.raises(GraphPatchError, match="convex"):
        select_region(_snapshot(nonconvex), ["node:0", "node:2"], binding_id="bad")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("network_output", "network output"),
        ("shape", "shape tensor"),
        ("host", "host tensor"),
        ("plugin", "unsupported layer"),
    ],
)
def test_selection_rejects_unsupported_boundaries(mutation: str, message: str) -> None:
    network, layers = _network()
    selected_id = "node:1"
    if mutation == "network_output":
        selected_id = "node:4"
    elif mutation == "shape":
        layers["replace_me"].get_output(0).is_shape_tensor = True
    elif mutation == "host":
        layers["replace_me"].get_output(0).location = "HOST"
    else:
        layers["replace_me"].type = "PLUGIN_V2"
    with pytest.raises(GraphPatchError, match=message):
        select_region(_snapshot(network), [selected_id], binding_id="bad")


def test_load_rejects_snapshot_tampering(tmp_path) -> None:
    network, _ = _network()
    snapshot = _snapshot(network)
    snapshot_doc = snapshot.to_dict()
    snapshot_doc["nodes"][0]["name"] = "tampered"
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot_doc))
    with pytest.raises(GraphPatchError, match="fingerprint"):
        load_snapshot(snapshot_path)

@pytest.mark.parametrize(
    ("binding_id", "extra_args", "message"),
    [
        ("bad/name", [], "binding_id"),
        ("valid", [{"type": "other"}], "type"),
        ("valid", [{"type": "int", "value": 1.5}], "integer"),
        ("valid", [{"type": "float", "value": float("inf")}], "finite"),
        ("valid", [{"type": "none", "value": 0}], "fields"),
    ],
)
def test_selection_rejects_invalid_binding_or_extra_args(
    binding_id: str,
    extra_args: list[dict],
    message: str,
) -> None:
    network, _ = _network()
    with pytest.raises(GraphPatchError, match=message):
        select_region(
            _snapshot(network),
            ["node:1"],
            binding_id=binding_id,
            extra_args=extra_args,
        )
