# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-independent graph snapshots and subgraph replacement helpers.

The objects consumed here deliberately use a small duck-typed surface:

* a network exposes ``num_layers``/``get_layer`` and, optionally,
  ``num_inputs``/``get_input`` plus ``num_outputs``/``get_output``;
* a layer exposes ``num_inputs``/``get_input``, ``num_outputs``/``get_output``,
  and ``set_input`` when it is rewired;
* a tensor may expose ``name``, ``dtype``, and ``shape``.

This makes graph selection unit-testable without importing TensorRT and keeps
selection artifacts independent of a TensorRT version.
The helper intentionally does not try to delete the selected layers.  It
rewires every consumer outside the selected region, leaving the old region
unreachable for the backend to prune.

Replacement callbacks may add backend layers while producing their outputs.
TensorRT does not expose a general rollback API, so callers must discard the
entire network if a callback or later validation raises.  The helper delays
consumer rewiring until every callback output has passed its ABI checks, but
does not claim transactional rollback of layers created by callbacks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


GRAPH_SNAPSHOT_SCHEMA_VERSION = 1
REGION_ARTIFACT_SCHEMA_VERSION = 1
GRAPH_REGION_SELECTION_SCHEMA_VERSION = 1
GRAPH_REGION_SELECTION_KIND = "tensorrt_model_connect.graph_region"
GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION = 1
GRAPH_REGION_SELECTION_SET_KIND = "tensorrt_model_connect.graph_region_set"
# Short aliases for callers that treat the set as the primary region artifact.
GRAPH_REGION_SET_SCHEMA_VERSION = GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION
GRAPH_REGION_SET_KIND = GRAPH_REGION_SELECTION_SET_KIND

ProvenanceProvider = (
    Mapping[Any, Mapping[str, Any]] | Callable[[Any, int], Mapping[str, Any] | None]
)
LayerIdentityProvider = Callable[[Any, int], Mapping[str, Any] | None]
ReplacementCallback = Callable[
    [Any, tuple[Any, ...], "RegionArtifact"],
    Sequence[Any],
]


class GraphPatchError(ValueError):
    """A graph cannot be snapshotted or safely patched."""


def _json_value(value: Any) -> Any:
    """Convert metadata and backend values into deterministic JSON values."""

    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise GraphPatchError("Graph metadata strings must be valid UTF-8") from exc
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphPatchError("Graph metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise GraphPatchError("Graph metadata object keys must be strings")
        return {
            key: _json_value(item) for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise GraphPatchError(
        "Graph metadata values must use JSON scalars, arrays, or objects; "
        f"got {type(value).__name__}"
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fingerprint(expected: Any, actual: str, artifact: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise GraphPatchError(
            f"{artifact} fingerprint must be a 64-character lowercase SHA-256 digest"
        )
    if expected != actual:
        raise GraphPatchError(
            f"{artifact} fingerprint mismatch: expected {expected}, computed {actual}"
        )


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    where: str,
) -> None:
    if not isinstance(value, Mapping):
        raise GraphPatchError(f"{where} must be a JSON object")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise GraphPatchError(f"{where} object keys must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise GraphPatchError(f"Unknown {where} fields: {sorted(unknown)}")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    _reject_unknown_fields(value, expected, where)
    missing = expected - set(value)
    if missing:
        raise GraphPatchError(f"Missing {where} fields: {sorted(missing)}")


def _require_string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise GraphPatchError(f"{where} must be {qualifier}")
    return value


def _require_optional_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, where)


def _require_array(value: Any, where: str) -> list[Any]:
    if type(value) is not list:
        raise GraphPatchError(f"{where} must be a JSON array")
    return value


def _require_string_array(
    value: Any,
    where: str,
    *,
    allow_null: bool = False,
) -> tuple[str | None, ...]:
    raw = _require_array(value, where)
    result: list[str | None] = []
    for index, item in enumerate(raw):
        if item is None and allow_null:
            result.append(None)
            continue
        result.append(_require_string(item, f"{where}[{index}]"))
    return tuple(result)


def _load_json_object(value: str | bytes, where: str) -> Mapping[str, Any]:
    """Decode one strict JSON object without silently collapsing duplicate keys."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GraphPatchError(f"{where} must be valid UTF-8: {exc}") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise GraphPatchError(f"{where} must be a string or UTF-8 bytes")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise GraphPatchError(f"{where} contains duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise GraphPatchError(f"{where} contains non-finite number {constant}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except GraphPatchError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise GraphPatchError(f"{where} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise GraphPatchError(f"{where} JSON must contain an object")
    return decoded


@dataclass(frozen=True)
class LayerIdentityContract:
    """Versioned model-owned meaning of per-layer identity attributes.

    ``provider_id`` identifies the model/family-owned provider implementation.
    Its owner must increment ``schema_version`` whenever the meaning or coverage
    of the returned attributes changes.
    """

    provider_id: str
    schema_version: int

    def validate(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise GraphPatchError("Layer identity provider_id must be a non-empty string")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise GraphPatchError("Layer identity schema_version must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "provider_id": self.provider_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LayerIdentityContract":
        _require_exact_fields(
            value,
            {"provider_id", "schema_version"},
            "layer identity contract",
        )
        contract = cls(
            provider_id=_require_string(
                value["provider_id"],
                "Layer identity contract provider_id",
            ),
            schema_version=value["schema_version"],
        )
        contract.validate()
        return contract


@dataclass(frozen=True)
class Tensor:
    """One tensor in a backend-neutral graph snapshot."""

    id: str
    name: str
    dtype: str | None = None
    shape: tuple[Any, ...] = ()
    producer: str | None = None
    consumers: tuple[str, ...] = ()
    is_shape_tensor: bool | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dtype": self.dtype,
            "shape": _json_value(self.shape),
            "producer": self.producer,
            "consumers": list(self.consumers),
            "is_shape_tensor": self.is_shape_tensor,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Tensor":
        _require_exact_fields(
            value,
            {
                "id",
                "name",
                "dtype",
                "shape",
                "producer",
                "consumers",
                "is_shape_tensor",
                "location",
            },
            "tensor",
        )
        shape = _require_array(value["shape"], "Tensor shape")
        for index, dimension in enumerate(shape):
            try:
                _json_value(dimension)
            except GraphPatchError as exc:
                raise GraphPatchError(f"Tensor shape[{index}] is invalid: {exc}") from exc
        consumers = _require_string_array(value["consumers"], "Tensor consumers")
        is_shape_tensor = value["is_shape_tensor"]
        if is_shape_tensor is not None and type(is_shape_tensor) is not bool:
            raise GraphPatchError("Tensor is_shape_tensor must be a boolean or null")
        return cls(
            id=_require_string(value["id"], "Tensor id"),
            name=_require_string(value["name"], "Tensor name", allow_empty=True),
            dtype=_require_optional_string(value["dtype"], "Tensor dtype"),
            shape=tuple(shape),
            producer=_require_optional_string(value["producer"], "Tensor producer"),
            consumers=consumers,
            is_shape_tensor=is_shape_tensor,
            location=_require_optional_string(value["location"], "Tensor location"),
        )


@dataclass(frozen=True)
class Node:
    """One layer/op in a backend-neutral graph snapshot."""

    id: str
    name: str
    op: str
    inputs: tuple[str | None, ...] = ()
    outputs: tuple[str | None, ...] = ()
    identity_attributes: Mapping[str, Any] = field(default_factory=dict)
    identity_complete: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "identity": {
                "attributes": _json_value(self.identity_attributes),
                "complete": self.identity_complete,
            },
            "provenance": _json_value(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Node":
        _require_exact_fields(
            value,
            {
                "id",
                "name",
                "op",
                "inputs",
                "outputs",
                "identity",
                "provenance",
            },
            "node",
        )
        raw_identity = value.get("identity", {})
        raw_provenance = value.get("provenance", {})
        if not isinstance(raw_identity, Mapping):
            raise GraphPatchError("Node identity must be a JSON object")
        _require_exact_fields(
            raw_identity,
            {"attributes", "complete"},
            "node identity",
        )
        raw_identity_attributes = raw_identity["attributes"]
        if not isinstance(raw_identity_attributes, Mapping):
            raise GraphPatchError("Node identity attributes must be a JSON object")
        identity_complete = raw_identity["complete"]
        if type(identity_complete) is not bool:
            raise GraphPatchError("Node identity complete must be a boolean")
        if not isinstance(raw_provenance, Mapping):
            raise GraphPatchError("Node provenance must be a JSON object")
        return cls(
            id=_require_string(value["id"], "Node id"),
            name=_require_string(value["name"], "Node name", allow_empty=True),
            op=_require_string(value["op"], "Node op"),
            inputs=_require_string_array(
                value["inputs"],
                "Node inputs",
                allow_null=True,
            ),
            outputs=_require_string_array(
                value["outputs"],
                "Node outputs",
                allow_null=True,
            ),
            identity_attributes=_json_value(raw_identity_attributes),
            identity_complete=identity_complete,
            provenance=_json_value(raw_provenance),
        )


@dataclass(frozen=True)
class GraphSnapshot:
    """A serializable, fingerprinted view of a network definition."""

    name: str
    nodes: tuple[Node, ...]
    tensors: tuple[Tensor, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    identity_contract: LayerIdentityContract | None = None
    schema_version: int = GRAPH_SNAPSHOT_SCHEMA_VERSION

    @property
    def identity_complete(self) -> bool:
        return self.identity_contract is not None and all(
            node.identity_complete for node in self.nodes
        )

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "contract": (
                None if self.identity_contract is None else self.identity_contract.to_dict()
            ),
            "complete": self.identity_complete,
        }

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": _json_value(self.metadata),
            "identity": self._identity_dict(),
        }

    def _fingerprint_payload_dict(self) -> dict[str, Any]:
        """Return graph identity without presentation-only provenance.

        Source frames are useful UI breadcrumbs, but the outer call site is
        necessarily different between ``graph inspect`` and a patched build.
        Structural identity therefore includes node names/types/connectivity,
        tensor contracts, graph I/O, and build metadata while excluding source
        files, functions, and line numbers.
        """

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "identity": {
                        "attributes": _json_value(node.identity_attributes),
                        "complete": node.identity_complete,
                    },
                }
                for node in self.nodes
            ],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": _json_value(self.metadata),
            "identity": self._identity_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._fingerprint_payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["fingerprint"] = self.fingerprint
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphSnapshot":
        _require_exact_fields(
            value,
            {
                "schema_version",
                "name",
                "nodes",
                "tensors",
                "inputs",
                "outputs",
                "metadata",
                "identity",
                "fingerprint",
            },
            "graph snapshot",
        )
        raw_metadata = value.get("metadata", {})
        raw_identity = value.get("identity", {})
        if not isinstance(raw_metadata, Mapping):
            raise GraphPatchError("Graph metadata must be a JSON object")
        if not isinstance(raw_identity, Mapping):
            raise GraphPatchError("Graph identity must be a JSON object")
        _require_exact_fields(
            raw_identity,
            {"contract", "complete"},
            "graph identity",
        )
        raw_contract = raw_identity["contract"]
        if raw_contract is not None and not isinstance(raw_contract, Mapping):
            raise GraphPatchError("Graph identity contract must be a JSON object or null")
        identity_complete = raw_identity["complete"]
        if type(identity_complete) is not bool:
            raise GraphPatchError("Graph identity complete must be a boolean")
        schema_version = value["schema_version"]
        if type(schema_version) is not int:
            raise GraphPatchError("Graph snapshot schema_version must be an integer")
        raw_nodes = _require_array(value["nodes"], "Graph snapshot nodes")
        raw_tensors = _require_array(value["tensors"], "Graph snapshot tensors")
        snapshot = cls(
            schema_version=schema_version,
            name=_require_string(
                value["name"],
                "Graph snapshot name",
                allow_empty=True,
            ),
            nodes=tuple(Node.from_dict(node) for node in raw_nodes),
            tensors=tuple(Tensor.from_dict(tensor) for tensor in raw_tensors),
            inputs=_require_string_array(
                value["inputs"],
                "Graph snapshot inputs",
            ),
            outputs=_require_string_array(
                value["outputs"],
                "Graph snapshot outputs",
            ),
            metadata=_json_value(raw_metadata),
            identity_contract=(
                None if raw_contract is None else LayerIdentityContract.from_dict(raw_contract)
            ),
        )
        if snapshot.schema_version != GRAPH_SNAPSHOT_SCHEMA_VERSION:
            raise GraphPatchError(
                "Unsupported graph snapshot schema_version "
                f"{snapshot.schema_version}; expected "
                f"{GRAPH_SNAPSHOT_SCHEMA_VERSION}"
            )
        snapshot._validate_identity()
        if identity_complete != snapshot.identity_complete:
            raise GraphPatchError(
                "Graph identity completeness does not match its node identity records"
            )
        _validate_fingerprint(
            value.get("fingerprint"),
            snapshot.fingerprint,
            "Graph snapshot",
        )
        snapshot._validate_references()
        return snapshot

    @classmethod
    def from_json(cls, value: str | bytes) -> "GraphSnapshot":
        return cls.from_dict(_load_json_object(value, "Graph snapshot"))

    def _validate_references(self) -> None:
        node_ids = [node.id for node in self.nodes]
        tensor_ids = [tensor.id for tensor in self.tensors]
        if len(node_ids) != len(set(node_ids)):
            raise GraphPatchError("Graph snapshot contains duplicate node IDs")
        if len(tensor_ids) != len(set(tensor_ids)):
            raise GraphPatchError("Graph snapshot contains duplicate tensor IDs")

        node_set = set(node_ids)
        tensor_set = set(tensor_ids)
        for node in self.nodes:
            unknown = {
                item
                for item in (*node.inputs, *node.outputs)
                if item is not None and item not in tensor_set
            }
            if unknown:
                raise GraphPatchError(
                    f"Node {node.id} references unknown tensors: {sorted(unknown)}"
                )
        for tensor in self.tensors:
            references = set(tensor.consumers)
            if tensor.producer is not None:
                references.add(tensor.producer)
            unknown = references - node_set
            if unknown:
                raise GraphPatchError(
                    f"Tensor {tensor.id} references unknown nodes: {sorted(unknown)}"
                )
        unknown_inputs = set(self.inputs) - tensor_set
        unknown_outputs = set(self.outputs) - tensor_set
        if unknown_inputs or unknown_outputs:
            raise GraphPatchError(
                "Graph inputs/outputs reference unknown tensors: "
                f"{sorted(unknown_inputs | unknown_outputs)}"
            )
        if len(self.inputs) != len(set(self.inputs)):
            raise GraphPatchError("Graph snapshot contains duplicate graph input tensor IDs")
        if len(self.outputs) != len(set(self.outputs)):
            raise GraphPatchError("Graph snapshot contains duplicate graph output tensor IDs")

        node_by_id = {node.id: node for node in self.nodes}
        tensor_by_id = {tensor.id: tensor for tensor in self.tensors}
        for node in self.nodes:
            for tensor_id in {item for item in node.inputs if item is not None}:
                expected_uses = sum(item == tensor_id for item in node.inputs)
                actual_uses = tensor_by_id[tensor_id].consumers.count(node.id)
                if actual_uses != expected_uses:
                    raise GraphPatchError(
                        f"Node {node.id} input references and tensor {tensor_id} "
                        "consumer references disagree"
                    )
            for tensor_id in {item for item in node.outputs if item is not None}:
                if tensor_by_id[tensor_id].producer != node.id:
                    raise GraphPatchError(
                        f"Node {node.id} output and tensor {tensor_id} producer disagree"
                    )
        for tensor in self.tensors:
            if tensor.producer is not None:
                producer = node_by_id[tensor.producer]
                if producer.outputs.count(tensor.id) != 1:
                    raise GraphPatchError(
                        f"Tensor {tensor.id} producer and node {producer.id} outputs disagree"
                    )
            for consumer_id in set(tensor.consumers):
                consumer = node_by_id[consumer_id]
                if consumer.inputs.count(tensor.id) != tensor.consumers.count(consumer_id):
                    raise GraphPatchError(
                        f"Tensor {tensor.id} consumers and node {consumer_id} inputs disagree"
                    )

    def _validate_identity(self) -> None:
        if self.identity_contract is not None:
            self.identity_contract.validate()
        elif any(node.identity_complete for node in self.nodes):
            raise GraphPatchError(
                "Complete node identity records require a layer identity contract"
            )
        for node in self.nodes:
            if type(node.identity_complete) is not bool:
                raise GraphPatchError(f"Node {node.id} identity completeness must be a boolean")
            if not isinstance(node.identity_attributes, Mapping):
                raise GraphPatchError(f"Node {node.id} identity attributes must be a mapping")
            _json_value(node.identity_attributes)


@dataclass(frozen=True)
class RegionBoundary:
    """The automatically inferred cut around selected nodes."""

    selected_node_ids: tuple[str, ...]
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    internal_tensor_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegionArtifact:
    """Portable contract produced by selecting a region in a graph snapshot."""

    graph_name: str
    graph_fingerprint: str
    selected_node_ids: tuple[str, ...]
    nodes: tuple[Node, ...]
    tensors: tuple[Tensor, ...]
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    internal_tensor_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REGION_ARTIFACT_SCHEMA_VERSION

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph": {
                "name": self.graph_name,
                "fingerprint": self.graph_fingerprint,
            },
            "selection": {
                "nodes": list(self.selected_node_ids),
            },
            "boundary": {
                "inputs": list(self.input_tensor_ids),
                "outputs": list(self.output_tensor_ids),
                "internal": list(self.internal_tensor_ids),
            },
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "metadata": _json_value(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["region_fingerprint"] = self.fingerprint
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegionArtifact":
        _require_exact_fields(
            value,
            {
                "schema_version",
                "graph",
                "selection",
                "boundary",
                "nodes",
                "tensors",
                "metadata",
                "region_fingerprint",
            },
            "region artifact",
        )
        graph = value["graph"]
        selection = value["selection"]
        boundary = value["boundary"]
        metadata = value["metadata"]
        if not all(isinstance(item, Mapping) for item in (graph, selection, boundary, metadata)):
            raise GraphPatchError("Region graph, selection, boundary, and metadata must be objects")
        _require_exact_fields(graph, {"name", "fingerprint"}, "region graph")
        _require_exact_fields(selection, {"nodes"}, "region selection")
        _require_exact_fields(
            boundary,
            {"inputs", "outputs", "internal"},
            "region boundary",
        )
        schema_version = value["schema_version"]
        if type(schema_version) is not int:
            raise GraphPatchError("Region artifact schema_version must be an integer")
        raw_nodes = _require_array(value["nodes"], "Region artifact nodes")
        raw_tensors = _require_array(value["tensors"], "Region artifact tensors")
        artifact = cls(
            schema_version=schema_version,
            graph_name=_require_string(
                graph["name"],
                "Region graph name",
                allow_empty=True,
            ),
            graph_fingerprint=_require_string(
                graph["fingerprint"],
                "Region graph fingerprint",
            ),
            selected_node_ids=_require_string_array(
                selection["nodes"],
                "Region selected nodes",
            ),
            nodes=tuple(Node.from_dict(node) for node in raw_nodes),
            tensors=tuple(Tensor.from_dict(tensor) for tensor in raw_tensors),
            input_tensor_ids=_require_string_array(
                boundary["inputs"],
                "Region boundary inputs",
            ),
            output_tensor_ids=_require_string_array(
                boundary["outputs"],
                "Region boundary outputs",
            ),
            internal_tensor_ids=_require_string_array(
                boundary["internal"],
                "Region boundary internal tensors",
            ),
            metadata=_json_value(metadata),
        )
        if artifact.schema_version != REGION_ARTIFACT_SCHEMA_VERSION:
            raise GraphPatchError(
                "Unsupported region artifact schema_version "
                f"{artifact.schema_version}; expected "
                f"{REGION_ARTIFACT_SCHEMA_VERSION}"
            )
        _validate_fingerprint(
            value["region_fingerprint"],
            artifact.fingerprint,
            "Region artifact",
        )
        return artifact

    @classmethod
    def from_json(cls, value: str | bytes) -> "RegionArtifact":
        return cls.from_dict(_load_json_object(value, "Region artifact"))


@dataclass(frozen=True)
class GraphRegionSelection:
    """Exact structural region selection pinned to one graph fingerprint."""

    graph_fingerprint: str
    selected_node_ids: tuple[str, ...]
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    model: str = ""
    stage: str = ""
    instance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = GRAPH_REGION_SELECTION_SCHEMA_VERSION
    kind: str = GRAPH_REGION_SELECTION_KIND

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphRegionSelection":
        allowed = {
            "schema_version",
            "kind",
            "graph",
            "selection",
            "boundary",
            "instance",
        }
        _reject_unknown_fields(value, allowed, "graph-region")
        missing = {"schema_version", "kind", "graph", "selection", "boundary"} - set(value)
        if missing:
            raise GraphPatchError(f"Missing graph-region fields: {sorted(missing)}")
        graph = value["graph"]
        selection = value["selection"]
        boundary = value["boundary"]
        if not all(isinstance(item, Mapping) for item in (graph, selection, boundary)):
            raise GraphPatchError("Region graph, selection, and boundary must be JSON objects")
        _require_exact_fields(
            graph,
            {"model", "stage", "fingerprint"},
            "graph-region graph",
        )
        _require_exact_fields(
            selection,
            {"node_ids"},
            "graph-region selection",
        )
        _require_exact_fields(
            boundary,
            {"inputs", "outputs"},
            "graph-region boundary",
        )

        def parse_tensor_refs(
            raw: Any,
            where: str,
        ) -> tuple[str, ...]:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
                raise GraphPatchError(f"{where} must be a JSON array")
            ordered: list[str] = []
            seen: set[str] = set()
            for index, reference in enumerate(raw):
                if not isinstance(reference, Mapping):
                    raise GraphPatchError(f"{where}[{index}] must be a JSON object")
                _require_exact_fields(
                    reference,
                    {"tensor_id"},
                    f"{where}[{index}]",
                )
                tensor_id = reference["tensor_id"]
                if not isinstance(tensor_id, str) or not tensor_id:
                    raise GraphPatchError(f"{where}[{index}].tensor_id must be a non-empty string")
                if tensor_id in seen:
                    raise GraphPatchError(f"{where} contains duplicate tensor ID {tensor_id}")
                seen.add(tensor_id)
                ordered.append(tensor_id)
            return tuple(ordered)

        raw_node_ids = selection["node_ids"]
        if not isinstance(raw_node_ids, Sequence) or isinstance(
            raw_node_ids, (str, bytes, bytearray)
        ):
            raise GraphPatchError("selection.node_ids must be a JSON array")
        if any(not isinstance(item, str) or not item for item in raw_node_ids):
            raise GraphPatchError("selection.node_ids must contain at least one non-empty node ID")
        selected_node_ids = tuple(raw_node_ids)
        if not selected_node_ids:
            raise GraphPatchError("selection.node_ids must contain at least one non-empty node ID")
        if len(selected_node_ids) != len(set(selected_node_ids)):
            raise GraphPatchError("selection.node_ids contains duplicate node IDs")

        fingerprint = graph["fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise GraphPatchError("graph.fingerprint must be a non-empty string")
        schema_version = value.get("schema_version")
        if type(schema_version) is not int or (
            schema_version != GRAPH_REGION_SELECTION_SCHEMA_VERSION
        ):
            raise GraphPatchError(
                "Unsupported graph-region schema_version "
                f"{schema_version!r}; expected "
                f"{GRAPH_REGION_SELECTION_SCHEMA_VERSION}"
            )
        kind = value.get("kind")
        if kind != GRAPH_REGION_SELECTION_KIND:
            raise GraphPatchError(
                f"Unsupported graph-region kind {kind!r}; expected {GRAPH_REGION_SELECTION_KIND!r}"
            )
        instance = value.get("instance", {})
        if not isinstance(instance, Mapping):
            raise GraphPatchError("instance must be a JSON object")
        model = graph["model"]
        stage = graph["stage"]
        if not isinstance(model, str):
            raise GraphPatchError("graph.model must be a string")
        if not isinstance(stage, str):
            raise GraphPatchError("graph.stage must be a string")
        input_tensor_ids = parse_tensor_refs(
            boundary["inputs"],
            "boundary.inputs",
        )
        output_tensor_ids = parse_tensor_refs(
            boundary["outputs"],
            "boundary.outputs",
        )

        return cls(
            schema_version=schema_version,
            kind=kind,
            graph_fingerprint=fingerprint,
            selected_node_ids=selected_node_ids,
            input_tensor_ids=input_tensor_ids,
            output_tensor_ids=output_tensor_ids,
            model=model,
            stage=stage,
            instance=_json_value(instance),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "GraphRegionSelection":
        return cls.from_dict(_load_json_object(value, "Graph-region"))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "graph": {
                "model": self.model,
                "stage": self.stage,
                "fingerprint": self.graph_fingerprint,
            },
            "selection": {
                "node_ids": list(self.selected_node_ids),
            },
            "boundary": {
                "inputs": [{"tensor_id": tensor_id} for tensor_id in self.input_tensor_ids],
                "outputs": [{"tensor_id": tensor_id} for tensor_id in self.output_tensor_ids],
            },
        }
        if self.instance:
            value["instance"] = _json_value(self.instance)
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def validate(self, snapshot: GraphSnapshot) -> RegionBoundary:
        """Recompute and validate the selected cut against ``snapshot``."""

        if type(self.schema_version) is not int or (
            self.schema_version != GRAPH_REGION_SELECTION_SCHEMA_VERSION
        ):
            raise GraphPatchError(
                "Unsupported graph-region schema_version "
                f"{self.schema_version!r}; expected "
                f"{GRAPH_REGION_SELECTION_SCHEMA_VERSION}"
            )
        if self.kind != GRAPH_REGION_SELECTION_KIND:
            raise GraphPatchError(
                f"Unsupported graph-region kind {self.kind!r}; "
                f"expected {GRAPH_REGION_SELECTION_KIND!r}"
            )
        if not isinstance(self.graph_fingerprint, str) or not self.graph_fingerprint:
            raise GraphPatchError("graph.fingerprint must be a non-empty string")
        if not isinstance(self.instance, Mapping):
            raise GraphPatchError("instance must be a mapping")
        if not isinstance(self.model, str) or not isinstance(self.stage, str):
            raise GraphPatchError("Region model and stage must be strings")
        if not self.selected_node_ids or len(self.selected_node_ids) != len(
            set(self.selected_node_ids)
        ):
            raise GraphPatchError("selection.node_ids must contain unique, non-empty node IDs")
        if any(not item for item in self.selected_node_ids):
            raise GraphPatchError("selection.node_ids must contain unique, non-empty node IDs")
        if len(self.input_tensor_ids) != len(set(self.input_tensor_ids)):
            raise GraphPatchError("Region input boundary contains duplicate tensor IDs")
        if len(self.output_tensor_ids) != len(set(self.output_tensor_ids)):
            raise GraphPatchError("Region output boundary contains duplicate tensor IDs")
        if any(not item for item in self.input_tensor_ids):
            raise GraphPatchError("Region input boundary contains an empty tensor ID")
        if any(not item for item in self.output_tensor_ids):
            raise GraphPatchError("Region output boundary contains an empty tensor ID")
        if snapshot.fingerprint != self.graph_fingerprint:
            raise GraphPatchError(
                "Selected graph fingerprint no longer matches the build graph: "
                f"expected {self.graph_fingerprint}, got {snapshot.fingerprint}"
            )
        boundary = validate_region(snapshot, self.selected_node_ids)
        if boundary.input_tensor_ids != self.input_tensor_ids:
            raise GraphPatchError(
                "Selected region input boundary no longer matches the build graph: "
                f"expected {self.input_tensor_ids}, got "
                f"{boundary.input_tensor_ids}"
            )
        if boundary.output_tensor_ids != self.output_tensor_ids:
            raise GraphPatchError(
                "Selected region output boundary no longer matches the build graph: "
                f"expected {self.output_tensor_ids}, got "
                f"{boundary.output_tensor_ids}"
            )
        return boundary


@dataclass(frozen=True)
class GraphRegionSelectionSet:
    """A set of independent region instances from one graph snapshot.

    Repeated transformer blocks are represented as separate regions instead of
    one disconnected selection.  Every region is validated against the same
    immutable graph snapshot and invokes the replacement callback once.

    ``from_dict`` and ``from_json`` also accept the original single-region
    structural document.
    """

    graph_fingerprint: str
    selections: tuple[GraphRegionSelection, ...]
    model: str = ""
    stage: str = ""
    build: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION
    kind: str = GRAPH_REGION_SELECTION_SET_KIND

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphRegionSelectionSet":
        kind = value.get("kind")
        if kind == GRAPH_REGION_SELECTION_KIND and "selection" in value:
            selection = GraphRegionSelection.from_dict(value)
            return cls(
                graph_fingerprint=selection.graph_fingerprint,
                selections=(selection,),
                model=selection.model,
                stage=selection.stage,
            )
        _require_exact_fields(
            value,
            {"schema_version", "kind", "graph", "regions"},
            "graph-region-set",
        )
        if kind != GRAPH_REGION_SELECTION_SET_KIND:
            raise GraphPatchError(
                "Unsupported graph-region-set kind "
                f"{kind!r}; expected {GRAPH_REGION_SELECTION_SET_KIND!r}"
            )

        schema_version = value.get("schema_version")
        if type(schema_version) is not int or (
            schema_version != GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION
        ):
            raise GraphPatchError(
                "Unsupported graph-region-set schema_version "
                f"{schema_version!r}; expected "
                f"{GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION}"
            )

        graph = value["graph"]
        if not isinstance(graph, Mapping):
            raise GraphPatchError("Region-set graph must be a JSON object")
        _reject_unknown_fields(
            graph,
            {"model", "stage", "fingerprint", "build"},
            "graph-region-set graph",
        )
        missing_graph = {"model", "stage", "fingerprint"} - set(graph)
        if missing_graph:
            raise GraphPatchError(f"Missing graph-region-set graph fields: {sorted(missing_graph)}")
        fingerprint = graph["fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise GraphPatchError("graph.fingerprint must be a non-empty string")
        model = graph["model"]
        stage = graph["stage"]
        if not isinstance(model, str):
            raise GraphPatchError("graph.model must be a string")
        if not isinstance(stage, str):
            raise GraphPatchError("graph.stage must be a string")
        build = graph.get("build", {})
        if not isinstance(build, Mapping):
            raise GraphPatchError("graph.build must be a JSON object")

        raw_regions = value["regions"]
        if not isinstance(raw_regions, Sequence) or isinstance(
            raw_regions,
            (str, bytes, bytearray),
        ):
            raise GraphPatchError("regions must be a JSON array")
        if not raw_regions:
            raise GraphPatchError("regions must contain at least one region")

        selections: list[GraphRegionSelection] = []
        for index, raw_region in enumerate(raw_regions):
            if not isinstance(raw_region, Mapping):
                raise GraphPatchError(f"regions[{index}] must be a JSON object")
            _reject_unknown_fields(
                raw_region,
                {"selection", "boundary", "instance"},
                f"regions[{index}]",
            )
            missing_region = {"selection", "boundary"} - set(raw_region)
            if missing_region:
                raise GraphPatchError(f"Missing regions[{index}] fields: {sorted(missing_region)}")
            region_document = {
                "schema_version": GRAPH_REGION_SELECTION_SCHEMA_VERSION,
                "kind": GRAPH_REGION_SELECTION_KIND,
                "graph": {
                    "model": model,
                    "stage": stage,
                    "fingerprint": fingerprint,
                },
                "selection": raw_region["selection"],
                "boundary": raw_region["boundary"],
                "instance": raw_region.get("instance", {}),
            }
            selection = GraphRegionSelection.from_dict(region_document)
            if selection.graph_fingerprint != fingerprint:
                raise GraphPatchError(
                    f"regions[{index}] fingerprint does not match the region-set graph fingerprint"
                )
            if selection.model and model and selection.model != model:
                raise GraphPatchError(
                    f"regions[{index}] model does not match the region-set graph model"
                )
            if selection.stage and stage and selection.stage != stage:
                raise GraphPatchError(
                    f"regions[{index}] stage does not match the region-set graph stage"
                )
            selections.append(selection)

        result = cls(
            schema_version=schema_version,
            kind=kind,
            graph_fingerprint=fingerprint,
            selections=tuple(selections),
            model=model,
            stage=stage,
            build=_json_value(build),
        )
        result._validate_static_independence()
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "GraphRegionSelectionSet":
        return cls.from_dict(_load_json_object(value, "Graph-region-set"))

    def to_dict(self) -> dict[str, Any]:
        graph: dict[str, Any] = {
            "model": self.model,
            "stage": self.stage,
            "fingerprint": self.graph_fingerprint,
        }
        if self.build:
            graph["build"] = _json_value(self.build)
        regions: list[dict[str, Any]] = []
        for selection in self.selections:
            region = selection.to_dict()
            regions.append(
                {
                    "selection": region["selection"],
                    "boundary": region["boundary"],
                    **({"instance": region["instance"]} if "instance" in region else {}),
                }
            )
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "graph": graph,
            "regions": regions,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def _validate_static_independence(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION
        ):
            raise GraphPatchError(
                "Unsupported graph-region-set schema_version "
                f"{self.schema_version!r}; expected "
                f"{GRAPH_REGION_SELECTION_SET_SCHEMA_VERSION}"
            )
        if self.kind != GRAPH_REGION_SELECTION_SET_KIND:
            raise GraphPatchError(
                "Unsupported graph-region-set kind "
                f"{self.kind!r}; expected "
                f"{GRAPH_REGION_SELECTION_SET_KIND!r}"
            )
        if not isinstance(self.graph_fingerprint, str) or not self.graph_fingerprint:
            raise GraphPatchError("A graph region set requires a non-empty graph fingerprint")
        if not isinstance(self.build, Mapping):
            raise GraphPatchError("Graph region-set build metadata must be a mapping")
        if not isinstance(self.model, str) or not isinstance(self.stage, str):
            raise GraphPatchError("Graph region-set model and stage must be strings")
        if not self.selections:
            raise GraphPatchError("A graph region set must contain at least one region")
        owner: dict[str, int] = {}
        for region_index, selection in enumerate(self.selections):
            if not isinstance(selection, GraphRegionSelection):
                raise GraphPatchError(f"Region {region_index} must be a GraphRegionSelection")
            if (
                selection.schema_version != GRAPH_REGION_SELECTION_SCHEMA_VERSION
                or selection.kind != GRAPH_REGION_SELECTION_KIND
            ):
                raise GraphPatchError(f"Region {region_index} has an unsupported schema or kind")
            if selection.graph_fingerprint != self.graph_fingerprint:
                raise GraphPatchError(
                    f"Region {region_index} fingerprint does not match "
                    "the region-set graph fingerprint"
                )
            if selection.model and self.model and selection.model != self.model:
                raise GraphPatchError(
                    f"Region {region_index} model does not match the region-set graph model"
                )
            if selection.stage and self.stage and selection.stage != self.stage:
                raise GraphPatchError(
                    f"Region {region_index} stage does not match the region-set graph stage"
                )
            for node_id in selection.selected_node_ids:
                previous = owner.get(node_id)
                if previous is not None:
                    raise GraphPatchError(
                        "Graph region-set instances must not overlap: "
                        f"node {node_id} appears in regions {previous} and "
                        f"{region_index}"
                    )
                owner[node_id] = region_index

    def validate(
        self,
        snapshot: GraphSnapshot,
    ) -> tuple[RegionBoundary, ...]:
        """Validate every independent instance against one original snapshot."""

        self._validate_static_independence()
        if snapshot.fingerprint != self.graph_fingerprint:
            raise GraphPatchError(
                "Selected graph fingerprint no longer matches the build graph: "
                f"expected {self.graph_fingerprint}, got {snapshot.fingerprint}"
            )

        boundaries = tuple(selection.validate(snapshot) for selection in self.selections)
        owner = {
            node_id: region_index
            for region_index, boundary in enumerate(boundaries)
            for node_id in boundary.selected_node_ids
        }
        for tensor in snapshot.tensors:
            producer_region = owner.get(tensor.producer)
            if producer_region is None:
                continue
            for consumer in tensor.consumers:
                consumer_region = owner.get(consumer)
                if consumer_region is not None and consumer_region != producer_region:
                    raise GraphPatchError(
                        "Graph region-set instances must not directly depend "
                        "on each other: "
                        f"{tensor.id} connects region {producer_region} "
                        f"to region {consumer_region}"
                    )
        return boundaries


def coerce_region_selection_set(
    selection: GraphRegionSelection | GraphRegionSelectionSet,
) -> GraphRegionSelectionSet:
    """Wrap the legacy single-region selection in a one-instance set."""

    if isinstance(selection, GraphRegionSelectionSet):
        return selection
    if not isinstance(selection, GraphRegionSelection):
        raise GraphPatchError(
            "Graph patch selection must be a GraphRegionSelection or GraphRegionSelectionSet"
        )
    return GraphRegionSelectionSet(
        graph_fingerprint=selection.graph_fingerprint,
        selections=(selection,),
        model=selection.model,
        stage=selection.stage,
    )


def _validate_selected_region(
    snapshot: GraphSnapshot,
    boundary: RegionBoundary,
) -> None:
    """Reject disconnected or stateful regions in the first prototype."""

    selected = set(boundary.selected_node_ids)
    adjacency = {node_id: set() for node_id in selected}
    for tensor in snapshot.tensors:
        if tensor.producer not in selected:
            continue
        for consumer in tensor.consumers:
            if consumer in selected:
                adjacency[tensor.producer].add(consumer)
                adjacency[consumer].add(tensor.producer)

    pending = [boundary.selected_node_ids[0]]
    visited: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(adjacency[node_id] - visited)
    if visited != selected:
        raise GraphPatchError(
            "The graph-patch prototype requires one connected region; "
            f"disconnected nodes: {sorted(selected - visited)}"
        )

    successors = {node.id: set() for node in snapshot.nodes}
    for tensor in snapshot.tensors:
        if tensor.producer is None:
            continue
        successors[tensor.producer].update(tensor.consumers)
    for start in boundary.selected_node_ids:
        pending_outside = [node_id for node_id in successors[start] if node_id not in selected]
        visited_outside: set[str] = set()
        while pending_outside:
            node_id = pending_outside.pop()
            if node_id in visited_outside:
                continue
            visited_outside.add(node_id)
            for successor in successors.get(node_id, ()):
                if successor in selected:
                    raise GraphPatchError(
                        "The graph-patch prototype requires a convex region; "
                        f"an unselected path connects {start} back to {successor}"
                    )
                if successor not in visited_outside:
                    pending_outside.append(successor)

    unsafe_tokens = (
        "ASSERTION",
        "CONDITION",
        "CONDITIONAL",
        "DIST_COLLECTIVE",
        "ITERATOR",
        "KV_CACHE_UPDATE",
        "LOOP",
        "PLUGIN",
        "RECURRENCE",
        "TRIP_LIMIT",
    )
    node_by_id = {node.id: node for node in snapshot.nodes}
    unsafe = [
        f"{node_id}:{node_by_id[node_id].op}"
        for node_id in boundary.selected_node_ids
        if any(token in node_by_id[node_id].op.upper() for token in unsafe_tokens)
    ]
    if unsafe:
        raise GraphPatchError(
            "The graph-patch prototype cannot replace stateful, control-flow, "
            f"collective, or existing plugin layers: {unsafe}"
        )


def validate_region(
    snapshot: GraphSnapshot,
    selected_node_ids: Sequence[str],
) -> RegionBoundary:
    """Derive a boundary and enforce structural replacement safety."""

    boundary = compute_region_boundary(snapshot, selected_node_ids)
    _validate_selected_region(snapshot, boundary)
    return boundary


@dataclass(frozen=True)
class RewireResult:
    """Receipt for a successful region replacement."""

    artifact: RegionArtifact
    replacement_outputs: tuple[Any, ...]
    rewired_consumer_inputs: int
    rewired_network_outputs: int


@dataclass(frozen=True)
class RewireBatchResult:
    """Aggregate receipt for replacing independent region instances."""

    results: tuple[RewireResult, ...]

    @property
    def region_count(self) -> int:
        return len(self.results)

    @property
    def selected_node_count(self) -> int:
        return sum(len(result.artifact.selected_node_ids) for result in self.results)

    @property
    def rewired_consumer_inputs(self) -> int:
        return sum(result.rewired_consumer_inputs for result in self.results)

    @property
    def rewired_network_outputs(self) -> int:
        return sum(result.rewired_network_outputs for result in self.results)

    @property
    def artifacts(self) -> tuple[RegionArtifact, ...]:
        return tuple(result.artifact for result in self.results)

    @property
    def artifact(self) -> RegionArtifact:
        """Preserve the historical accessor for one-region callers."""

        if len(self.results) != 1:
            raise GraphPatchError("A multi-region result has no single artifact; use artifacts")
        return self.results[0].artifact


@dataclass
class _TensorDraft:
    id: str
    value: Any
    name: str
    dtype: str | None
    shape: tuple[Any, ...]
    is_shape_tensor: bool | None
    location: str | None
    producer: str | None = None
    consumers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CapturedGraph:
    """One graph snapshot plus identity-safe live-handle lookup."""

    snapshot: GraphSnapshot
    _layers: Mapping[str, Any] = field(repr=False)
    _tensors: Mapping[str, Any] = field(repr=False)
    _node_ids_by_identity: Mapping[int, tuple[str, ...]] = field(repr=False)
    _tensor_ids_by_identity: Mapping[int, str] = field(repr=False)
    _node_ids_by_signature: Mapping[
        tuple[
            str,
            str,
            tuple[str | None, ...],
            tuple[str | None, ...],
        ],
        tuple[str, ...],
    ] = field(repr=False)

    def node_id_for(self, layer: Any) -> str:
        identity_candidates = self._node_ids_by_identity.get(id(layer), ())
        if len(identity_candidates) == 1 and self._layers[identity_candidates[0]] is layer:
            return identity_candidates[0]
        # TensorRT can expose one C++ layer through a typed wrapper returned by
        # add_*() and a distinct generic ILayer wrapper from get_layer().  Its
        # ordered tensor handles are stable across those wrappers.
        try:
            signature = _layer_signature(layer, self.tensor_id_for)
        except (
            AttributeError,
            GraphPatchError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            signature = None
        candidates = () if signature is None else self._node_ids_by_signature.get(signature, ())
        if len(candidates) != 1:
            raise GraphPatchError("Layer handle does not belong to this captured graph")
        return candidates[0]

    def tensor_id_for(self, tensor: Any) -> str:
        tensor_id = self._tensor_ids_by_identity.get(id(tensor))
        if tensor_id is None or self._tensors[tensor_id] is not tensor:
            raise GraphPatchError("Tensor handle does not belong to this captured graph")
        return tensor_id

    def _layer_for(self, node_id: str) -> Any:
        return self._layers[node_id]

    def _tensor_for(self, tensor_id: str) -> Any:
        return self._tensors[tensor_id]

    def _original_tensor_id(self, tensor: Any) -> str | None:
        tensor_id = self._tensor_ids_by_identity.get(id(tensor))
        if tensor_id is None or self._tensors[tensor_id] is not tensor:
            return None
        return tensor_id


class _TensorRegistry:
    """Give repeated backend tensor handles deterministic snapshot IDs."""

    def __init__(self) -> None:
        self._by_identity: dict[int, str] = {}
        # Strong references prevent object-id reuse during one capture.
        self._values: list[Any] = []
        self.drafts: list[_TensorDraft] = []

    def register(self, tensor: Any) -> str:
        known = self._by_identity.get(id(tensor))
        if known is not None:
            return known

        tensor_id = f"tensor:{len(self.drafts)}"
        self._values.append(tensor)
        self._by_identity[id(tensor)] = tensor_id
        self.drafts.append(
            _TensorDraft(
                id=tensor_id,
                value=tensor,
                name=_object_name(tensor, tensor_id),
                dtype=_tensor_dtype(tensor),
                shape=_tensor_shape(tensor),
                is_shape_tensor=_optional_tensor_bool(
                    tensor,
                    "is_shape_tensor",
                ),
                location=_tensor_location(tensor),
            )
        )
        return tensor_id

    def draft(self, tensor_id: str) -> _TensorDraft:
        return self.drafts[int(tensor_id.split(":", 1)[1])]


def _count(value: Any, attribute: str, *, required: bool = False) -> int:
    raw = getattr(value, attribute, None)
    if raw is None:
        if required:
            raise GraphPatchError(f"Object does not expose required {attribute}")
        return 0
    if callable(raw):
        raw = raw()
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise GraphPatchError(f"{attribute} must be an integer, got {raw!r}") from exc


def _layer_signature(
    layer: Any,
    tensor_id_for: Callable[[Any], str],
) -> tuple[
    str,
    str,
    tuple[str | None, ...],
    tuple[str | None, ...],
]:
    name = getattr(layer, "name", None)
    raw_op = getattr(layer, "type", None)
    if raw_op is None:
        raw_op = getattr(layer, "op", type(layer).__name__)

    def ids(method_name: str, count_name: str) -> tuple[str | None, ...]:
        method = getattr(layer, method_name, None)
        count = _count(layer, count_name, required=True)
        if not callable(method):
            raise GraphPatchError(f"Layer does not expose required {method_name}")
        result: list[str | None] = []
        for index in range(count):
            tensor = method(index)
            result.append(None if tensor is None else tensor_id_for(tensor))
        return tuple(result)

    return (
        "" if name is None else str(name),
        str(raw_op),
        ids("get_input", "num_inputs"),
        ids("get_output", "num_outputs"),
    )


def _object_name(value: Any, fallback: str) -> str:
    name = getattr(value, "name", None)
    if name is None:
        return fallback
    text = str(name)
    return text if text else fallback


def _tensor_dtype(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return None
    text = str(dtype)
    normalized = text.lower().replace("_", "")
    if normalized in {"float16", "half", "datatype.half"}:
        return "float16"
    if normalized in {"bfloat16", "bf16", "datatype.bf16"}:
        return "bfloat16"
    if normalized in {"float32", "float", "datatype.float"}:
        return "float32"
    return text


def _tensor_shape(tensor: Any) -> tuple[Any, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(_json_value(dimension) for dimension in shape)
    except TypeError:
        return (str(shape),)


def _optional_tensor_bool(tensor: Any, attribute: str) -> bool | None:
    try:
        value = getattr(tensor, attribute)
    except (AttributeError, RuntimeError):
        return None
    if value is None:
        return None
    return bool(value)


def _tensor_location(tensor: Any) -> str | None:
    try:
        value = getattr(tensor, "location")
    except (AttributeError, RuntimeError):
        return None
    return None if value is None else str(value)


def _node_provenance(
    provider: ProvenanceProvider | None,
    layer: Any,
    index: int,
    node_id: str,
    node_name: str,
) -> Mapping[str, Any]:
    if provider is None:
        return {}
    if callable(provider):
        raw = provider(layer, index)
    else:
        raw = None
        for key in (node_id, node_name, index, str(index)):
            if key in provider:
                raw = provider[key]
                break
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise GraphPatchError(
            f"Provenance for {node_id} must be a mapping, got {type(raw).__name__}"
        )
    return _json_value(raw)


def _node_identity(
    provider: LayerIdentityProvider | None,
    layer: Any,
    index: int,
    node_id: str,
) -> tuple[Mapping[str, Any], bool]:
    if provider is None:
        return {}, False
    raw = provider(layer, index)
    if raw is None:
        return {}, False
    if not isinstance(raw, Mapping):
        raise GraphPatchError(
            f"Layer identity for {node_id} must be a mapping or None, got {type(raw).__name__}"
        )
    return _json_value(raw), True


def capture_network(
    network: Any,
    *,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: ProvenanceProvider | None = None,
    identity_provider: LayerIdentityProvider | None = None,
    identity_contract: LayerIdentityContract | None = None,
) -> CapturedGraph:
    """Capture a graph and retain live object-to-ID bindings by identity.

    The model builder owns ``identity_provider``.  For every layer it must
    return a deterministic JSON object covering every layer-specific setting
    that can change execution semantics.  TensorRT's generic layer wrapper may
    hide subtype attributes, so providers may need to record those settings
    when the model creates the layer.  Returning ``None`` marks that layer's
    identity incomplete: the snapshot remains inspectable but cannot be used
    for rewiring.
    """
    if (identity_provider is None) != (identity_contract is None):
        raise GraphPatchError(
            "Layer identity provider and contract must either both be present or both be absent"
        )
    if identity_provider is not None and not callable(identity_provider):
        raise GraphPatchError("Layer identity provider must be callable")
    if identity_contract is not None:
        if not isinstance(identity_contract, LayerIdentityContract):
            raise GraphPatchError("Layer identity contract has the wrong type")
        identity_contract.validate()

    layer_count = _count(network, "num_layers", required=True)
    layers = [network.get_layer(index) for index in range(layer_count)]
    registry = _TensorRegistry()

    graph_inputs: list[str] = []
    for index in range(_count(network, "num_inputs")):
        tensor = network.get_input(index)
        if tensor is not None:
            graph_inputs.append(registry.register(tensor))

    # Register outputs in graph order first so tensor IDs are stable by producer.
    layer_output_ids: list[tuple[str | None, ...]] = []
    for index, layer in enumerate(layers):
        node_id = f"node:{index}"
        output_ids: list[str | None] = []
        for output_index in range(_count(layer, "num_outputs")):
            tensor = layer.get_output(output_index)
            if tensor is None:
                output_ids.append(None)
                continue
            tensor_id = registry.register(tensor)
            draft = registry.draft(tensor_id)
            if draft.producer is not None and draft.producer != node_id:
                raise GraphPatchError(
                    f"Tensor {tensor_id} has multiple producers: {draft.producer} and {node_id}"
                )
            draft.producer = node_id
            output_ids.append(tensor_id)
        layer_output_ids.append(tuple(output_ids))

    nodes: list[Node] = []
    layer_bindings: dict[str, Any] = {}
    for index, layer in enumerate(layers):
        node_id = f"node:{index}"
        layer_bindings[node_id] = layer
        input_ids: list[str | None] = []
        for input_index in range(_count(layer, "num_inputs")):
            tensor = layer.get_input(input_index)
            if tensor is None:
                input_ids.append(None)
                continue
            tensor_id = registry.register(tensor)
            registry.draft(tensor_id).consumers.append(node_id)
            input_ids.append(tensor_id)

        node_name = _object_name(layer, node_id)
        raw_op = getattr(layer, "type", None)
        if raw_op is None:
            raw_op = getattr(layer, "op", type(layer).__name__)
        identity_attributes, identity_complete = _node_identity(
            identity_provider,
            layer,
            index,
            node_id,
        )
        nodes.append(
            Node(
                id=node_id,
                name=node_name,
                op=str(raw_op),
                inputs=tuple(input_ids),
                outputs=layer_output_ids[index],
                identity_attributes=identity_attributes,
                identity_complete=identity_complete,
                provenance=_node_provenance(
                    provenance,
                    layer,
                    index,
                    node_id,
                    node_name,
                ),
            )
        )

    graph_outputs: list[str] = []
    for index in range(_count(network, "num_outputs")):
        tensor = network.get_output(index)
        if tensor is not None:
            graph_outputs.append(registry.register(tensor))

    tensors = tuple(
        Tensor(
            id=draft.id,
            name=draft.name,
            dtype=draft.dtype,
            shape=draft.shape,
            producer=draft.producer,
            consumers=tuple(draft.consumers),
            is_shape_tensor=draft.is_shape_tensor,
            location=draft.location,
        )
        for draft in registry.drafts
    )
    snapshot = GraphSnapshot(
        name=(str(name) if name is not None else _object_name(network, "forward")),
        nodes=tuple(nodes),
        tensors=tensors,
        inputs=tuple(graph_inputs),
        outputs=tuple(graph_outputs),
        metadata=_json_value(metadata or {}),
        identity_contract=identity_contract,
    )
    snapshot._validate_references()
    snapshot._validate_identity()
    tensor_bindings = {draft.id: draft.value for draft in registry.drafts}
    tensor_ids_by_identity = {
        id(tensor): tensor_id for tensor_id, tensor in tensor_bindings.items()
    }

    def captured_tensor_id(tensor: Any) -> str:
        tensor_id = tensor_ids_by_identity.get(id(tensor))
        if tensor_id is None or tensor_bindings[tensor_id] is not tensor:
            raise GraphPatchError("Layer signature references an uncaptured tensor")
        return tensor_id

    signature_owners: dict[
        tuple[
            str,
            str,
            tuple[str | None, ...],
            tuple[str | None, ...],
        ],
        list[str],
    ] = {}
    for node_id, layer in layer_bindings.items():
        signature = _layer_signature(layer, captured_tensor_id)
        signature_owners.setdefault(signature, []).append(node_id)

    identity_owners: dict[int, list[str]] = {}
    for node_id, layer in layer_bindings.items():
        identity_owners.setdefault(id(layer), []).append(node_id)

    return CapturedGraph(
        snapshot=snapshot,
        _layers=layer_bindings,
        _tensors=tensor_bindings,
        _node_ids_by_identity={
            identity: tuple(node_ids) for identity, node_ids in identity_owners.items()
        },
        _tensor_ids_by_identity=tensor_ids_by_identity,
        _node_ids_by_signature={
            signature: tuple(node_ids) for signature, node_ids in signature_owners.items()
        },
    )


def snapshot_network(
    network: Any,
    *,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: ProvenanceProvider | None = None,
    identity_provider: LayerIdentityProvider | None = None,
    identity_contract: LayerIdentityContract | None = None,
) -> GraphSnapshot:
    """Snapshot an ``INetworkDefinition``-like object without importing TRT.

    ``provenance`` can be either a ``(layer, index) -> mapping`` callback or a
    mapping keyed by node ID, layer name, integer index, or string index.  This
    is the hook through which builder scopes/source locations can later be
    attached to diagnostic graph artifacts.

    ``identity_provider`` and ``identity_contract`` are an all-or-nothing,
    model-owned pair.  See :func:`capture_network` for their completeness
    contract.  Omitting them intentionally produces a structural-only snapshot
    suitable for inspection and selection, but not rewiring.
    """

    return capture_network(
        network,
        name=name,
        metadata=metadata,
        provenance=provenance,
        identity_provider=identity_provider,
        identity_contract=identity_contract,
    ).snapshot


def compute_region_boundary(
    snapshot: GraphSnapshot,
    selected_node_ids: Sequence[str],
) -> RegionBoundary:
    """Infer the ordered graph cut around ``selected_node_ids``."""

    selected_requested = set(selected_node_ids)
    if not selected_requested:
        raise GraphPatchError("A graph region must select at least one node")
    all_node_ids = {node.id for node in snapshot.nodes}
    unknown = selected_requested - all_node_ids
    if unknown:
        raise GraphPatchError(f"Selected unknown node IDs: {sorted(unknown)}")

    selected_nodes = tuple(node for node in snapshot.nodes if node.id in selected_requested)
    selected = {node.id for node in selected_nodes}
    tensor_by_id = {tensor.id: tensor for tensor in snapshot.tensors}
    graph_outputs = set(snapshot.outputs)

    input_ids: list[str] = []
    output_ids: list[str] = []
    internal_ids: list[str] = []
    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()
    seen_internal: set[str] = set()

    for node in selected_nodes:
        for tensor_id in node.inputs:
            if tensor_id is None or tensor_id in seen_inputs:
                continue
            if tensor_by_id[tensor_id].producer not in selected:
                seen_inputs.add(tensor_id)
                input_ids.append(tensor_id)

    for node in selected_nodes:
        for tensor_id in node.outputs:
            if tensor_id is None:
                continue
            tensor = tensor_by_id[tensor_id]
            is_output = tensor_id in graph_outputs or any(
                consumer not in selected for consumer in tensor.consumers
            )
            if is_output and tensor_id not in seen_outputs:
                seen_outputs.add(tensor_id)
                output_ids.append(tensor_id)
            elif not is_output and tensor_id not in seen_internal:
                seen_internal.add(tensor_id)
                internal_ids.append(tensor_id)

    return RegionBoundary(
        selected_node_ids=tuple(node.id for node in selected_nodes),
        input_tensor_ids=tuple(input_ids),
        output_tensor_ids=tuple(output_ids),
        internal_tensor_ids=tuple(internal_ids),
    )


def create_region_artifact(
    snapshot: GraphSnapshot,
    selected_node_ids: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> RegionArtifact:
    """Create a portable contract for a structurally safe region."""

    boundary = validate_region(snapshot, selected_node_ids)
    selected = set(boundary.selected_node_ids)
    nodes = tuple(node for node in snapshot.nodes if node.id in selected)
    relevant_tensor_ids = {
        tensor_id
        for node in nodes
        for tensor_id in (*node.inputs, *node.outputs)
        if tensor_id is not None
    }
    tensors = tuple(tensor for tensor in snapshot.tensors if tensor.id in relevant_tensor_ids)
    return RegionArtifact(
        graph_name=snapshot.name,
        graph_fingerprint=snapshot.fingerprint,
        selected_node_ids=boundary.selected_node_ids,
        nodes=nodes,
        tensors=tensors,
        input_tensor_ids=boundary.input_tensor_ids,
        output_tensor_ids=boundary.output_tensor_ids,
        internal_tensor_ids=boundary.internal_tensor_ids,
        metadata=_json_value(metadata or {}),
    )


def _capture_live_for_rewire(
    network: Any,
    snapshot: GraphSnapshot,
    *,
    current_name: str,
    current_metadata: Mapping[str, Any],
    identity_provider: LayerIdentityProvider,
    identity_contract: LayerIdentityContract,
    provenance: ProvenanceProvider | None,
) -> CapturedGraph:
    """Recapture independently supplied current identity before any callback."""

    snapshot._validate_identity()
    if not snapshot.identity_complete:
        raise GraphPatchError(
            "Graph rewiring requires a complete model-owned layer identity provider; "
            "the selected snapshot is structural-only or incomplete"
        )
    if not isinstance(current_name, str):
        raise GraphPatchError("Current graph name must be a string")
    if not isinstance(current_metadata, Mapping):
        raise GraphPatchError("Current graph metadata must be a mapping")
    if not isinstance(identity_contract, LayerIdentityContract):
        raise GraphPatchError("Current layer identity contract has the wrong type")
    identity_contract.validate()
    if identity_contract != snapshot.identity_contract:
        raise GraphPatchError(
            "Current layer identity contract does not match the selected graph snapshot"
        )
    if not callable(identity_provider):
        raise GraphPatchError("Current layer identity provider must be callable")

    live = capture_network(
        network,
        name=current_name,
        metadata=current_metadata,
        provenance=provenance,
        identity_provider=identity_provider,
        identity_contract=identity_contract,
    )
    if not live.snapshot.identity_complete:
        raise GraphPatchError(
            "Current layer identity provider did not completely describe every live layer"
        )
    if live.snapshot.fingerprint != snapshot.fingerprint:
        raise GraphPatchError(
            "Live graph no longer matches the selected graph snapshot: "
            f"expected {snapshot.fingerprint}, got {live.snapshot.fingerprint}"
        )
    return live


def rewire_region(
    network: Any,
    snapshot: GraphSnapshot,
    selected_node_ids: Sequence[str],
    replacement: ReplacementCallback,
    *,
    current_name: str,
    current_metadata: Mapping[str, Any],
    identity_provider: LayerIdentityProvider,
    identity_contract: LayerIdentityContract,
    provenance: ProvenanceProvider | None = None,
) -> RewireResult:
    """Replace a selected region by rewiring all of its external uses.

    The callback receives ``(network, boundary_inputs, region_artifact)`` and
    returns one newly created live backend tensor per ordered boundary output;
    returning a tensor from the original graph is rejected.  The helper
    validates the snapshot against the live network before calling it, rewires
    all external consumer slots with ``set_input``, and updates marked network
    outputs through ``unmark_output``/``mark_output`` when needed.

    ``current_name``, ``current_metadata``, and the identity provider/contract
    must describe the current build independently.  They are never replayed
    from ``snapshot``.  A structural-only or incompletely identified snapshot
    is rejected before ``replacement`` is called.

    If this function raises after invoking ``replacement``, discard ``network``;
    callback-created layers cannot be rolled back generically.
    """

    live = _capture_live_for_rewire(
        network,
        snapshot,
        current_name=current_name,
        current_metadata=current_metadata,
        identity_provider=identity_provider,
        identity_contract=identity_contract,
        provenance=provenance,
    )

    validate_region(snapshot, selected_node_ids)
    artifact = create_region_artifact(snapshot, selected_node_ids)
    return _rewire_captured_regions(
        network,
        snapshot,
        live,
        (artifact,),
        replacement,
    ).results[0]


@dataclass(frozen=True)
class _RewirePlan:
    artifact: RegionArtifact
    boundary_inputs: tuple[Any, ...]
    external_slots: tuple[tuple[Any, int, int], ...]
    marked_output_ids: frozenset[str]


def _try_set_tensor_name(tensor: Any, name: str) -> bool:
    try:
        tensor.name = name
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return getattr(tensor, "name", None) == name


def _preserve_replaced_output_name(
    snapshot: GraphSnapshot,
    tensor_id: str,
    old_output: Any,
    new_output: Any,
) -> None:
    """Move a public output name off the old tensor before reusing it."""

    old_name = getattr(old_output, "name", None)
    if old_output is new_output or old_name is None:
        return
    if not hasattr(new_output, "name"):
        raise GraphPatchError(
            f"Replacement for network output {tensor_id} does not expose a writable name"
        )

    stem = f"__trtmc_replaced_{snapshot.fingerprint[:12]}_{tensor_id.replace(':', '_')}"
    for suffix in range(100):
        temporary_name = stem if suffix == 0 else f"{stem}_{suffix}"
        if _try_set_tensor_name(old_output, temporary_name):
            break
    else:
        raise GraphPatchError(f"Could not reserve the public name of network output {tensor_id}")

    # Some bindings can return two Python wrappers for one backend tensor.
    # In that case the temporary rename is visible through both wrappers.
    if getattr(new_output, "name", None) == temporary_name:
        if not _try_set_tensor_name(new_output, old_name):
            raise GraphPatchError(
                f"Could not restore the public name of network output {tensor_id}"
            )
        return

    if _try_set_tensor_name(new_output, old_name):
        return

    restored = _try_set_tensor_name(old_output, old_name)
    suffix = "" if restored else "; the original name could not be restored"
    raise GraphPatchError(
        f"Could not preserve the public name of network output {tensor_id}{suffix}"
    )


def _validate_replacement_output(
    artifact: RegionArtifact,
    output_index: int,
    output: Any,
    *,
    region_index: int,
) -> None:
    contract_by_id = {tensor.id: tensor for tensor in artifact.tensors}
    tensor_id = artifact.output_tensor_ids[output_index]
    contract = contract_by_id[tensor_id]
    actual = {
        "dtype": _tensor_dtype(output),
        "shape": _tensor_shape(output),
        "location": _tensor_location(output),
        "is_shape_tensor": _optional_tensor_bool(output, "is_shape_tensor"),
    }
    expected = {
        "dtype": contract.dtype,
        "shape": contract.shape,
        "location": contract.location,
        "is_shape_tensor": contract.is_shape_tensor,
    }
    mismatches = [
        f"{field}: expected {wanted!r}, got {actual[field]!r}"
        for field, wanted in expected.items()
        if wanted is not None and actual[field] != wanted
    ]
    if mismatches:
        raise GraphPatchError(
            "Replacement output ABI mismatch for "
            f"region {region_index} output {output_index} ({tensor_id}): " + "; ".join(mismatches)
        )


def _rewire_captured_regions(
    network: Any,
    snapshot: GraphSnapshot,
    live: CapturedGraph,
    artifacts: Sequence[RegionArtifact],
    replacement: ReplacementCallback,
) -> RewireBatchResult:
    """Plan and apply all rewires against one pre-callback live capture."""

    plans: list[_RewirePlan] = []
    for artifact in artifacts:
        selected = set(artifact.selected_node_ids)
        output_index_by_id = {
            tensor_id: index for index, tensor_id in enumerate(artifact.output_tensor_ids)
        }
        external_slots: list[tuple[Any, int, int]] = []
        for node in snapshot.nodes:
            if node.id in selected:
                continue
            for input_index, input_tensor_id in enumerate(node.inputs):
                output_index = output_index_by_id.get(input_tensor_id)
                if output_index is None:
                    continue
                layer = live._layer_for(node.id)
                if not callable(getattr(layer, "set_input", None)):
                    raise GraphPatchError(f"External consumer {node.id} does not expose set_input")
                external_slots.append((layer, input_index, output_index))

        marked_output_ids = frozenset(snapshot.outputs) & frozenset(artifact.output_tensor_ids)
        if marked_output_ids and (
            not callable(getattr(network, "unmark_output", None))
            or not callable(getattr(network, "mark_output", None))
        ):
            raise GraphPatchError(
                "Selected region feeds a network output, but the network does "
                "not expose unmark_output/mark_output"
            )
        plans.append(
            _RewirePlan(
                artifact=artifact,
                boundary_inputs=tuple(
                    live._tensor_for(tensor_id) for tensor_id in artifact.input_tensor_ids
                ),
                external_slots=tuple(external_slots),
                marked_output_ids=marked_output_ids,
            )
        )

    # Invoke every replacement and validate every output contract before
    # rewiring the original graph.  A failing callback therefore cannot leave
    # only a prefix of consumer slots rewired.
    outputs_by_plan: list[tuple[Any, ...]] = []
    for region_index, plan in enumerate(plans):
        raw_outputs = replacement(
            network,
            plan.boundary_inputs,
            plan.artifact,
        )
        try:
            replacement_outputs = tuple(raw_outputs)
        except TypeError as exc:
            raise GraphPatchError(
                f"Replacement for region {region_index} did not return an output sequence"
            ) from exc
        if len(replacement_outputs) != len(plan.artifact.output_tensor_ids):
            raise GraphPatchError(
                "Replacement returned "
                f"{len(replacement_outputs)} outputs for region "
                f"{region_index}, which has "
                f"{len(plan.artifact.output_tensor_ids)} boundary outputs"
            )
        if any(output is None for output in replacement_outputs):
            raise GraphPatchError(
                f"Replacement outputs for region {region_index} cannot contain None"
            )
        for output_index, output in enumerate(replacement_outputs):
            existing_tensor_id = live._original_tensor_id(output)
            if existing_tensor_id is not None:
                raise GraphPatchError(
                    "Replacement outputs must be newly created backend tensors; "
                    f"region {region_index} output {output_index} aliases "
                    f"original tensor {existing_tensor_id}"
                )
            _validate_replacement_output(
                plan.artifact,
                output_index,
                output,
                region_index=region_index,
            )
        outputs_by_plan.append(replacement_outputs)

    replacement_output_owners: dict[int, tuple[int, int]] = {}
    for region_index, replacement_outputs in enumerate(outputs_by_plan):
        for output_index, replacement_output in enumerate(replacement_outputs):
            previous = replacement_output_owners.get(id(replacement_output))
            if previous is not None:
                raise GraphPatchError(
                    "One replacement tensor cannot satisfy two region outputs: "
                    f"region {previous[0]} output {previous[1]} and "
                    f"region {region_index} output {output_index}"
                )
            replacement_output_owners[id(replacement_output)] = (
                region_index,
                output_index,
            )

    replacement_by_output_id: dict[str, Any] = {}
    for plan, replacement_outputs in zip(plans, outputs_by_plan):
        for tensor_id, replacement_output in zip(
            plan.artifact.output_tensor_ids,
            replacement_outputs,
        ):
            if tensor_id in plan.marked_output_ids:
                replacement_by_output_id[tensor_id] = replacement_output

    for tensor_id, replacement_output in replacement_by_output_id.items():
        _preserve_replaced_output_name(
            snapshot,
            tensor_id,
            live._tensor_for(tensor_id),
            replacement_output,
        )

    for plan, replacement_outputs in zip(plans, outputs_by_plan):
        for layer, input_index, output_index in plan.external_slots:
            layer.set_input(input_index, replacement_outputs[output_index])

    # TensorRT only exposes append-style mark_output(). Preserve binding order
    # by rebuilding the complete output list whenever one selected region
    # replaces a marked graph output.
    if replacement_by_output_id:
        for tensor_id in snapshot.outputs:
            network.unmark_output(live._tensor_for(tensor_id))
        for tensor_id in snapshot.outputs:
            old_output = live._tensor_for(tensor_id)
            new_output = replacement_by_output_id.get(tensor_id, old_output)
            network.mark_output(new_output)

    results: list[RewireResult] = []
    for plan, replacement_outputs in zip(plans, outputs_by_plan):
        results.append(
            RewireResult(
                artifact=plan.artifact,
                replacement_outputs=replacement_outputs,
                rewired_consumer_inputs=len(plan.external_slots),
                rewired_network_outputs=len(plan.marked_output_ids),
            )
        )
    return RewireBatchResult(results=tuple(results))


def rewire_selection(
    network: Any,
    snapshot: GraphSnapshot,
    selection: GraphRegionSelection,
    replacement: ReplacementCallback,
    *,
    current_name: str,
    current_metadata: Mapping[str, Any],
    identity_provider: LayerIdentityProvider,
    identity_contract: LayerIdentityContract,
    provenance: ProvenanceProvider | None = None,
) -> RewireResult:
    """Validate and replace one fully identified selection.

    Current graph identity is recaptured from the explicit ``current_*`` and
    identity arguments before ``replacement`` can run.
    """

    selection.validate(snapshot)
    return rewire_selection_set(
        network,
        snapshot,
        coerce_region_selection_set(selection),
        replacement,
        current_name=current_name,
        current_metadata=current_metadata,
        identity_provider=identity_provider,
        identity_contract=identity_contract,
        provenance=provenance,
    ).results[0]


def rewire_selection_set(
    network: Any,
    snapshot: GraphSnapshot,
    selection_set: GraphRegionSelectionSet,
    replacement: ReplacementCallback,
    *,
    current_name: str,
    current_metadata: Mapping[str, Any],
    identity_provider: LayerIdentityProvider,
    identity_contract: LayerIdentityContract,
    provenance: ProvenanceProvider | None = None,
) -> RewireBatchResult:
    """Replace every independent region against the same original snapshot.

    Each callback result must contain newly created, mutually distinct backend
    tensors; aliases to the original graph or another region output are
    rejected before consumer rewiring.

    Current graph identity is recaptured from the explicit ``current_*`` and
    identity arguments.  A structural-only or incompletely identified snapshot
    is rejected before ``replacement`` is called.

    If this function raises after invoking ``replacement``, discard ``network``;
    callback-created layers cannot be rolled back generically.
    """

    selection_set.validate(snapshot)
    live = _capture_live_for_rewire(
        network,
        snapshot,
        current_name=current_name,
        current_metadata=current_metadata,
        identity_provider=identity_provider,
        identity_contract=identity_contract,
        provenance=provenance,
    )
    artifacts = tuple(
        create_region_artifact(
            snapshot,
            selection.selected_node_ids,
            metadata=(
                {"instance": _json_value(selection.instance)} if selection.instance else None
            ),
        )
        for selection in selection_set.selections
    )
    return _rewire_captured_regions(
        network,
        snapshot,
        live,
        artifacts,
        replacement,
    )
