# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal structural TensorRT graph capture, selection, and rewiring."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence


class GraphPatchError(ValueError):
    """The selected graph region is invalid, unsafe, or stale."""


@dataclass(frozen=True)
class Tensor:
    id: str
    name: str
    dtype: str
    shape: tuple[int | str, ...]
    producer: str | None
    consumers: tuple[str, ...]
    is_shape_tensor: bool | None
    location: str | None


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    op: str
    inputs: tuple[str | None, ...]
    outputs: tuple[str | None, ...]


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: tuple[Node, ...]
    tensors: tuple[Tensor, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any]
    fingerprint: str

    def _payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("fingerprint")
        return {
            "schema_version": 1,
            "kind": "tensorrt_model_connect.graph_snapshot",
            **payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Any) -> "GraphSnapshot":
        root = _exact(
            value,
            "snapshot",
            "schema_version kind nodes tensors inputs outputs metadata fingerprint",
        )
        if root["schema_version"] != 1 or root["kind"] != ("tensorrt_model_connect.graph_snapshot"):
            raise GraphPatchError("invalid graph snapshot schema")
        nodes = []
        for value in _list(root["nodes"], "nodes"):
            item = _exact(value, "node", "id name op inputs outputs")
            nodes.append(
                Node(
                    _str(item["id"], "node.id"),
                    _str(item["name"], "node.name", allow_empty=True),
                    _str(item["op"], "node.op"),
                    _optional_strings(item["inputs"], "node.inputs"),
                    _optional_strings(item["outputs"], "node.outputs"),
                )
            )
        tensors = []
        for value in _list(root["tensors"], "tensors"):
            item = _exact(
                value,
                "tensor",
                "id name dtype shape producer consumers is_shape_tensor location",
            )
            shape = tuple(_dimension(value) for value in _list(item["shape"], "tensor.shape"))
            is_shape = item["is_shape_tensor"]
            if is_shape is not None and type(is_shape) is not bool:
                raise GraphPatchError("tensor.is_shape_tensor must be boolean or null")
            tensors.append(
                Tensor(
                    _str(item["id"], "tensor.id"),
                    _str(item["name"], "tensor.name", allow_empty=True),
                    _str(item["dtype"], "tensor.dtype"),
                    shape,
                    _optional_str(item["producer"], "tensor.producer"),
                    _strings(item["consumers"], "tensor.consumers", unique=False),
                    is_shape,
                    _optional_str(item["location"], "tensor.location", allow_empty=True),
                )
            )
        snapshot = cls(
            tuple(nodes),
            tuple(tensors),
            _strings(root["inputs"], "inputs"),
            _strings(root["outputs"], "outputs"),
            _json_object(root["metadata"], "metadata"),
            _str(root["fingerprint"], "fingerprint"),
        )
        _validate_snapshot(snapshot)
        actual = _hash(snapshot._payload())
        if snapshot.fingerprint != actual:
            raise GraphPatchError(
                f"snapshot fingerprint mismatch: expected {snapshot.fingerprint}, got {actual}"
            )
        return snapshot


@dataclass(frozen=True)
class RegionSelection:
    graph_fingerprint: str
    engine_role: str
    binding_id: str
    abi_sha256: str
    workspace_bytes: int
    extra_args: tuple[dict[str, Any], ...]
    node_ids: tuple[str, ...]
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    output_shape_input: int | None

    @property
    def kernel_name(self) -> str:
        return f"trtmc.slot.{self.binding_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Any) -> "RegionSelection":
        root = _exact(
            value,
            "selection",
            (
                "schema_version graph_fingerprint engine_role binding_id "
                "abi_sha256 workspace_bytes extra_args node_ids input_tensor_ids "
                "output_tensor_ids output_shape_input"
            ),
        )
        if root["schema_version"] != 1:
            raise GraphPatchError("selection schema_version must be 1")
        workspace = root["workspace_bytes"]
        if type(workspace) is not int or not 0 <= workspace <= (1 << 31) - 1:
            raise GraphPatchError("workspace_bytes must be a non-negative 32-bit integer")
        output_shape_input = root["output_shape_input"]
        if output_shape_input is not None and (
            type(output_shape_input) is not int or output_shape_input < 0
        ):
            raise GraphPatchError("output_shape_input must be a non-negative integer or null")
        selection = cls(
            _str(root["graph_fingerprint"], "graph_fingerprint"),
            _str(root["engine_role"], "engine_role"),
            _binding_id(root["binding_id"]),
            _sha256(root["abi_sha256"], "abi_sha256"),
            workspace,
            tuple(
                _extra_arg(item, f"extra_args[{index}]")
                for index, item in enumerate(_list(root["extra_args"], "extra_args"))
            ),
            _strings(root["node_ids"], "node_ids"),
            _strings(root["input_tensor_ids"], "input_tensor_ids"),
            _strings(root["output_tensor_ids"], "output_tensor_ids"),
            output_shape_input,
        )
        return selection


@dataclass(frozen=True)
class RewireResult:
    replacement_outputs: tuple[Any, ...]
    rewired_consumer_inputs: int


@dataclass
class _Draft:
    id: str
    value: Any
    name: str
    dtype: str
    shape: tuple[int | str, ...]
    is_shape_tensor: bool | None
    location: str | None
    producer: str | None = None
    consumers: list[str] | None = None


@dataclass(frozen=True)
class _Captured:
    snapshot: GraphSnapshot
    layers: dict[str, Any]
    tensors: dict[str, Any]
    tensor_ids: dict[int, str]
    attentions: dict[str, Any]


def _exact(value: Any, where: str, field_text: str) -> dict[str, Any]:
    fields = set(field_text.split())
    if type(value) is not dict or set(value) != fields:
        raise GraphPatchError(f"{where} fields must be exactly: {', '.join(sorted(fields))}")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if type(value) is not list:
        raise GraphPatchError(f"{where} must be an array")
    return value


def _str(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise GraphPatchError(f"{where} must be a string")
    return value


def _optional_str(value: Any, where: str, *, allow_empty: bool = False) -> str | None:
    return None if value is None else _str(value, where, allow_empty=allow_empty)


def _strings(value: Any, where: str, *, unique: bool = True) -> tuple[str, ...]:
    result = tuple(_str(item, where) for item in _list(value, where))
    if unique and len(set(result)) != len(result):
        raise GraphPatchError(f"{where} contains a duplicate")
    return result


def _optional_strings(value: Any, where: str) -> tuple[str | None, ...]:
    return tuple(None if item is None else _str(item, where) for item in _list(value, where))


def _dimension(value: Any) -> int | str:
    if type(value) not in (int, str):
        raise GraphPatchError("tensor shape dimensions must be integers or strings")
    return value


def _json_object(value: Any, where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GraphPatchError(f"{where} must be an object")
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise GraphPatchError(f"{where} must contain JSON values") from exc


def _binding_id(value: Any) -> str:
    value = _str(value, "binding_id")
    if re.fullmatch(r"[A-Za-z0-9_.@-]+", value) is None:
        raise GraphPatchError("binding_id may contain only A-Z, a-z, 0-9, _, ., @, and -")
    return value


def _sha256(value: Any, where: str) -> str:
    value = _str(value, where)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise GraphPatchError(f"{where} must be a lowercase SHA-256")
    return value


def _extra_arg(value: Any, where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GraphPatchError(f"{where} must be an object")
    arg_type = value.get("type")
    if arg_type in ("none", "ptr"):
        _exact(value, where, "type")
        return {"type": arg_type}
    if arg_type == "int":
        _exact(value, where, "type value")
        if type(value["value"]) is not int or not -(1 << 31) <= value["value"] < (1 << 31):
            raise GraphPatchError(f"{where}.value must be a signed 32-bit integer")
        return {"type": "int", "value": value["value"]}
    if arg_type == "float":
        _exact(value, where, "type value")
        if type(value["value"]) not in (int, float) or not math.isfinite(value["value"]):
            raise GraphPatchError(f"{where}.value must be finite numeric")
        return {"type": "float", "value": float(value["value"])}
    raise GraphPatchError(f"{where}.type must be none, int, float, or ptr")


def _hash(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _abi_hash(
    snapshot: GraphSnapshot,
    inputs: Sequence[str],
    outputs: Sequence[str],
    workspace_bytes: int,
    extra_args: Sequence[Mapping[str, Any]],
    output_shape_input: int | None,
) -> str:
    tensors = {tensor.id: tensor for tensor in snapshot.tensors}

    def specs(tensor_ids: Sequence[str]) -> list[dict[str, Any]]:
        return [
            {"dtype": tensors[tensor_id].dtype, "shape": list(tensors[tensor_id].shape)}
            for tensor_id in tensor_ids
        ]

    return _hash(
        {
            "format": "linear",
            "inputs": specs(inputs),
            "outputs": specs(outputs),
            "workspace_bytes": workspace_bytes,
            "extra_args": list(extra_args),
            "output_shape_input": output_shape_input,
        }
    )


def _count(value: Any, name: str) -> int:
    count = getattr(value, name, None)
    if type(count) is not int or count < 0:
        raise GraphPatchError(f"TensorRT object has invalid {name}")
    return count


def _shape(tensor: Any) -> tuple[int | str, ...]:
    result: list[int | str] = []
    for dimension in getattr(tensor, "shape", ()):
        try:
            result.append(int(dimension))
        except (TypeError, ValueError, OverflowError):
            result.append(str(dimension))
    return tuple(result)


_ATTENTION_INPUT_PROPERTIES = (
    "mask",
    "normalization_quantize_scale",
    "query_lengths",
    "key_value_lengths",
)


def _layer_inputs(
    layer: Any, attentions: Sequence[Any]
) -> tuple[tuple[Any | None, ...], Any | None]:
    count = _count(layer, "num_inputs")
    if "ATTENTION_INPUT" not in str(getattr(layer, "type", "")).upper():
        return tuple(layer.get_input(index) for index in range(count)), None
    generic = tuple(layer.get_input(index) for index in range(3))
    owner = next(
        (
            attention
            for attention in attentions
            if all(attention.get_input(index) is generic[index] for index in range(3))
        ),
        None,
    )
    if owner is None:
        if count > 3:
            raise GraphPatchError(
                "cannot capture typed IAttention inputs created outside Model Connect"
            )
        return generic, None
    typed = tuple(getattr(owner, name, None) for name in _ATTENTION_INPUT_PROPERTIES)
    return generic + typed, owner


def _set_layer_input(layer: Any, attention: Any | None, index: int, tensor: Any) -> None:
    if attention is not None and index >= 3:
        setattr(attention, _ATTENTION_INPUT_PROPERTIES[index - 3], tensor)
        return
    setter = getattr(layer, "set_input", None)
    if not callable(setter):
        raise GraphPatchError("external consumer cannot be rewired")
    setter(index, tensor)


def _capture(
    network: Any,
    metadata: Mapping[str, Any] | None,
    attentions: Sequence[Any] = (),
) -> _Captured:
    metadata_copy = _json_object(dict(metadata or {}), "metadata")
    if not isinstance(metadata_copy.get("engine_role"), str) or not metadata_copy["engine_role"]:
        raise GraphPatchError("snapshot metadata.engine_role must be a non-empty string")
    layers = [network.get_layer(index) for index in range(_count(network, "num_layers"))]
    drafts: list[_Draft] = []
    tensor_ids: dict[int, str] = {}

    def register(tensor: Any) -> str:
        identity = id(tensor)
        found = tensor_ids.get(identity)
        if found is not None and drafts[int(found[7:])].value is tensor:
            return found
        tensor_id = f"tensor:{len(drafts)}"
        tensor_ids[identity] = tensor_id
        is_shape = getattr(tensor, "is_shape_tensor", None)
        location = getattr(tensor, "location", None)
        drafts.append(
            _Draft(
                tensor_id,
                tensor,
                str(getattr(tensor, "name", "") or ""),
                str(getattr(tensor, "dtype", "unknown")),
                _shape(tensor),
                is_shape if type(is_shape) is bool else None,
                None if location is None else str(location),
                consumers=[],
            )
        )
        return tensor_id

    graph_inputs = tuple(
        register(network.get_input(index)) for index in range(_count(network, "num_inputs"))
    )
    layer_outputs: list[tuple[str | None, ...]] = []
    for index, layer in enumerate(layers):
        outputs = []
        for output_index in range(_count(layer, "num_outputs")):
            output = layer.get_output(output_index)
            if output is None:
                outputs.append(None)
                continue
            tensor_id = register(output)
            draft = drafts[int(tensor_id[7:])]
            if draft.producer is not None:
                raise GraphPatchError(f"{tensor_id} has multiple producers")
            draft.producer = f"node:{index}"
            outputs.append(tensor_id)
        layer_outputs.append(tuple(outputs))
    nodes = []
    live_layers = {}
    live_attentions = {}
    for index, layer in enumerate(layers):
        node_id = f"node:{index}"
        live_layers[node_id] = layer
        inputs = []
        layer_inputs, attention = _layer_inputs(layer, attentions)
        if attention is not None:
            live_attentions[node_id] = attention
        for tensor in layer_inputs:
            if tensor is None:
                inputs.append(None)
                continue
            tensor_id = register(tensor)
            drafts[int(tensor_id[7:])].consumers.append(node_id)
            inputs.append(tensor_id)
        nodes.append(
            Node(
                node_id,
                str(getattr(layer, "name", "") or ""),
                str(getattr(layer, "type", type(layer).__name__)),
                tuple(inputs),
                layer_outputs[index],
            )
        )
    graph_outputs = tuple(
        register(network.get_output(index)) for index in range(_count(network, "num_outputs"))
    )
    tensors = tuple(
        Tensor(
            draft.id,
            draft.name,
            draft.dtype,
            draft.shape,
            draft.producer,
            tuple(draft.consumers or ()),
            draft.is_shape_tensor,
            draft.location,
        )
        for draft in drafts
    )
    partial = GraphSnapshot(tuple(nodes), tensors, graph_inputs, graph_outputs, metadata_copy, "")
    snapshot = GraphSnapshot(
        partial.nodes,
        partial.tensors,
        partial.inputs,
        partial.outputs,
        partial.metadata,
        _hash(partial._payload()),
    )
    _validate_snapshot(snapshot)
    return _Captured(
        snapshot,
        live_layers,
        {draft.id: draft.value for draft in drafts},
        tensor_ids,
        live_attentions,
    )


def _validate_snapshot(snapshot: GraphSnapshot) -> None:
    nodes = {node.id for node in snapshot.nodes}
    tensors = {tensor.id for tensor in snapshot.tensors}
    if len(nodes) != len(snapshot.nodes) or len(tensors) != len(snapshot.tensors):
        raise GraphPatchError("snapshot contains duplicate IDs")
    for node in snapshot.nodes:
        if any(item is not None and item not in tensors for item in node.inputs + node.outputs):
            raise GraphPatchError(f"{node.id} references an unknown tensor")
    for tensor in snapshot.tensors:
        if tensor.producer is not None and tensor.producer not in nodes:
            raise GraphPatchError(f"{tensor.id} has an unknown producer")
        if any(consumer not in nodes for consumer in tensor.consumers):
            raise GraphPatchError(f"{tensor.id} has an unknown consumer")
    if any(item not in tensors for item in snapshot.inputs + snapshot.outputs):
        raise GraphPatchError("snapshot graph IO references an unknown tensor")


def snapshot_network(
    network: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    attentions: Sequence[Any] = (),
) -> GraphSnapshot:
    """Capture a raw TensorRT network in deterministic build order."""
    return _capture(network, metadata, attentions).snapshot


def _boundary(
    snapshot: GraphSnapshot, requested_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not requested_ids or any(type(item) is not str or not item for item in requested_ids):
        raise GraphPatchError("node_ids must be a non-empty string sequence")
    selected = set(requested_ids)
    if len(selected) != len(requested_ids):
        raise GraphPatchError("node_ids contains a duplicate")
    node_map = {node.id: node for node in snapshot.nodes}
    missing = sorted(selected - set(node_map))
    if missing:
        raise GraphPatchError("unknown selected node(s): " + ", ".join(missing))
    ordered = tuple(node.id for node in snapshot.nodes if node.id in selected)
    tensor_map = {tensor.id: tensor for tensor in snapshot.tensors}
    adjacency = {node_id: set() for node_id in ordered}
    successors = {node.id: set() for node in snapshot.nodes}
    for tensor in snapshot.tensors:
        if tensor.producer is None:
            continue
        successors[tensor.producer].update(tensor.consumers)
        if tensor.producer in selected:
            for consumer in tensor.consumers:
                if consumer in selected:
                    adjacency[tensor.producer].add(consumer)
                    adjacency[consumer].add(tensor.producer)
    pending, visited = [ordered[0]], set()
    while pending:
        node_id = pending.pop()
        if node_id not in visited:
            visited.add(node_id)
            pending.extend(adjacency[node_id] - visited)
    if visited != selected:
        raise GraphPatchError("selected region must be connected")
    for start in ordered:
        pending, outside = [x for x in successors[start] if x not in selected], set()
        while pending:
            node_id = pending.pop()
            if node_id in outside:
                continue
            outside.add(node_id)
            for successor in successors[node_id]:
                if successor in selected:
                    raise GraphPatchError("selected region must be convex")
                pending.append(successor)
    inputs: list[str] = []
    outputs: list[str] = []
    touched: set[str] = set()
    unsafe = (
        "ASSERTION",
        "CONDITION",
        "CONDITIONAL",
        "DIST_COLLECTIVE",
        "ITERATOR",
        "LOOP",
        "PLUGIN",
        "RECURRENCE",
        "TRIP_LIMIT",
    )
    for node_id in ordered:
        node = node_map[node_id]
        if any(token in node.op.upper() for token in unsafe):
            raise GraphPatchError(f"selected region contains unsupported layer {node_id}")
        touched.update(item for item in node.inputs + node.outputs if item is not None)
        for tensor_id in node.inputs:
            if tensor_id is not None and tensor_map[tensor_id].producer not in selected:
                if tensor_id not in inputs:
                    inputs.append(tensor_id)
        for tensor_id in node.outputs:
            if tensor_id is not None:
                tensor = tensor_map[tensor_id]
                is_boundary = tensor_id in snapshot.outputs or any(
                    consumer not in selected for consumer in tensor.consumers
                )
                if is_boundary and tensor_id not in outputs:
                    outputs.append(tensor_id)
    if not inputs or not outputs:
        raise GraphPatchError("selected region must have input and output boundaries")
    if len(outputs) != 1:
        raise GraphPatchError("selected region must have exactly one output boundary")
    if set(outputs) & set(snapshot.outputs):
        raise GraphPatchError("selected region cannot replace a network output")
    for tensor_id in touched:
        tensor = tensor_map[tensor_id]
        if tensor.is_shape_tensor:
            raise GraphPatchError(f"selected region touches shape tensor {tensor_id}")
        if tensor.location is not None and "HOST" in tensor.location.upper():
            raise GraphPatchError(f"selected region touches host tensor {tensor_id}")
    return ordered, tuple(inputs), tuple(outputs)


def _validate_output_shape_input(
    snapshot: GraphSnapshot,
    inputs: Sequence[str],
    outputs: Sequence[str],
    output_shape_input: int | None,
) -> None:
    tensors = {tensor.id: tensor for tensor in snapshot.tensors}
    output = tensors[outputs[0]]
    dynamic_output = not all(type(dim) is int and dim > 0 for dim in output.shape)
    if dynamic_output:
        if type(output_shape_input) is not int or not 0 <= output_shape_input < len(inputs):
            raise GraphPatchError(
                "dynamic output requires --output-shape-like-input with a boundary input index"
            )
        source = tensors[inputs[output_shape_input]]
        if (source.dtype, source.shape) != (output.dtype, output.shape):
            raise GraphPatchError("output shape input must have the output dtype and shape")
    elif output_shape_input is not None:
        raise GraphPatchError("fixed output does not use --output-shape-like-input")


def select_region(
    snapshot: GraphSnapshot,
    node_ids: Sequence[str],
    *,
    binding_id: str,
    workspace_bytes: int = 0,
    extra_args: Sequence[Mapping[str, Any]] = (),
    output_shape_input: int | None = None,
) -> RegionSelection:
    """Validate one explicit connected region and derive its ordered ABI."""
    binding_id = _binding_id(binding_id)
    if type(workspace_bytes) is not int or not 0 <= workspace_bytes <= (1 << 31) - 1:
        raise GraphPatchError("workspace_bytes must be a non-negative 32-bit integer")
    engine_role = _str(snapshot.metadata.get("engine_role"), "metadata.engine_role")
    ordered, inputs, outputs = _boundary(snapshot, node_ids)
    _validate_output_shape_input(snapshot, inputs, outputs, output_shape_input)
    canonical_args = tuple(
        _extra_arg(item, f"extra_args[{index}]") for index, item in enumerate(extra_args)
    )
    return RegionSelection(
        snapshot.fingerprint,
        engine_role,
        binding_id,
        _abi_hash(
            snapshot,
            inputs,
            outputs,
            workspace_bytes,
            canonical_args,
            output_shape_input,
        ),
        workspace_bytes,
        canonical_args,
        ordered,
        inputs,
        outputs,
        output_shape_input,
    )


def _validate_selection(snapshot: GraphSnapshot, selection: RegionSelection) -> None:
    if snapshot.fingerprint != selection.graph_fingerprint:
        raise GraphPatchError(
            "live graph fingerprint does not match selection: "
            f"expected {selection.graph_fingerprint}, got {snapshot.fingerprint}"
        )
    if snapshot.metadata["engine_role"] != selection.engine_role:
        raise GraphPatchError("live graph engine role does not match selection")
    nodes, inputs, outputs = _boundary(snapshot, selection.node_ids)
    if (nodes, inputs, outputs) != (
        selection.node_ids,
        selection.input_tensor_ids,
        selection.output_tensor_ids,
    ):
        raise GraphPatchError("selection boundary does not match the live graph")
    _validate_output_shape_input(snapshot, inputs, outputs, selection.output_shape_input)
    actual_abi = _abi_hash(
        snapshot,
        inputs,
        outputs,
        selection.workspace_bytes,
        selection.extra_args,
        selection.output_shape_input,
    )
    if actual_abi != selection.abi_sha256:
        raise GraphPatchError("selection ABI hash does not match the live graph")


def _contract(tensor: Any) -> tuple[Any, ...]:
    is_shape = getattr(tensor, "is_shape_tensor", None)
    location = getattr(tensor, "location", None)
    return (
        str(getattr(tensor, "dtype", "unknown")),
        _shape(tensor),
        is_shape if type(is_shape) is bool else None,
        None if location is None else str(location),
    )


def apply_region(
    network: Any,
    selection: RegionSelection,
    replacement: Callable[[Any, tuple[Any, ...], RegionSelection], Sequence[Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    attentions: Sequence[Any] = (),
) -> RewireResult:
    """Validate and replace one selected region before TRT serialization."""
    captured = _capture(network, metadata, attentions)
    _validate_selection(captured.snapshot, selection)
    selected = set(selection.node_ids)
    output_indexes = {
        tensor_id: index for index, tensor_id in enumerate(selection.output_tensor_ids)
    }
    slots = []
    for node in captured.snapshot.nodes:
        if node.id in selected:
            continue
        for input_index, tensor_id in enumerate(node.inputs):
            output_index = output_indexes.get(tensor_id)
            if output_index is not None:
                layer = captured.layers[node.id]
                slots.append(
                    (
                        layer,
                        captured.attentions.get(node.id),
                        input_index,
                        output_index,
                    )
                )
    live_inputs = tuple(captured.tensors[item] for item in selection.input_tensor_ids)
    outputs = tuple(replacement(network, live_inputs, selection))
    if len(outputs) != len(selection.output_tensor_ids) or any(x is None for x in outputs):
        raise GraphPatchError(
            f"replacement must return {len(selection.output_tensor_ids)} non-null outputs"
        )
    if len({id(output) for output in outputs}) != len(outputs):
        raise GraphPatchError("replacement outputs must be distinct tensors")
    tensor_map = {tensor.id: tensor for tensor in captured.snapshot.tensors}
    for index, output in enumerate(outputs):
        if id(output) in captured.tensor_ids:
            raise GraphPatchError("replacement outputs must be newly created tensors")
        expected = tensor_map[selection.output_tensor_ids[index]]
        actual = _contract(output)
        if actual[:2] != (expected.dtype, expected.shape):
            raise GraphPatchError(f"replacement output {index} ABI mismatch")
        if expected.is_shape_tensor is not None and actual[2] != expected.is_shape_tensor:
            raise GraphPatchError(f"replacement output {index} shape-tensor mismatch")
        if expected.location is not None and actual[3] != expected.location:
            raise GraphPatchError(f"replacement output {index} location mismatch")
    for layer, attention, input_index, output_index in slots:
        _set_layer_input(layer, attention, input_index, outputs[output_index])
    return RewireResult(outputs, len(slots))


def _read(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GraphPatchError(f"cannot read {label} {path}: {exc}") from exc


def _write(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(path: str | Path) -> GraphSnapshot:
    return GraphSnapshot.from_dict(_read(path, "snapshot"))


def write_snapshot(snapshot: GraphSnapshot, path: str | Path) -> None:
    _write(path, snapshot.to_dict())


def load_selection(path: str | Path) -> RegionSelection:
    return RegionSelection.from_dict(_read(path, "selection"))


def write_selection(selection: RegionSelection, path: str | Path) -> None:
    _write(path, selection.to_dict())
