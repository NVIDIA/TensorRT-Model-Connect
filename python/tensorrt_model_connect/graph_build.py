# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-scoped capture and application of one explicit TensorRT graph slot."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .graph_patch import (
    GraphPatchError,
    RegionSelection,
    apply_region,
    load_selection,
    snapshot_network,
    write_snapshot,
)


class GraphInspectionComplete(RuntimeError):
    """Internal success signal used to stop before TensorRT compilation."""


@dataclass
class _Session:
    mode: str
    role: str
    metadata: dict[str, Any]
    path: Path | None = None
    selection: RegionSelection | None = None
    completed: bool = False
    slot: dict[str, str] | None = None
    attentions: dict[int, list[Any]] = field(default_factory=dict)


_ACTIVE: ContextVar[_Session | None] = ContextVar("trtmc_graph_build", default=None)
_ROLE: ContextVar[str] = ContextVar("trtmc_graph_engine_role", default="")


def _activate(session: _Session):
    if _ACTIVE.get() is not None:
        raise GraphPatchError("graph build sessions cannot be nested")
    return _ACTIVE.set(session)


def _validate_role(role: str, metadata: Mapping[str, Any]) -> None:
    layout = metadata.get("decoder_engine_layout", "split")
    valid = ("dual_profile",) if layout == "dual_profile" else ("prefill", "decode")
    if role not in valid:
        raise GraphPatchError(
            f"engine role {role!r} is incompatible with decoder layout {layout!r}"
        )


@contextmanager
def inspect_graph(
    path: str | Path,
    *,
    engine_role: str,
    metadata: Mapping[str, Any],
) -> Iterator[None]:
    _validate_role(engine_role, metadata)
    session = _Session("inspect", engine_role, dict(metadata), path=Path(path))
    token = _activate(session)
    try:
        yield
        raise GraphPatchError(f"engine role {engine_role!r} was not built")
    finally:
        _ACTIVE.reset(token)


@contextmanager
def apply_graph_slot(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> Iterator[None]:
    selection = load_selection(path)
    _validate_role(selection.engine_role, metadata)
    session = _Session("apply", selection.engine_role, dict(metadata), selection=selection)
    token = _activate(session)
    try:
        yield
        if not session.completed:
            raise GraphPatchError(f"engine role {selection.engine_role!r} was not built")
    finally:
        _ACTIVE.reset(token)


@contextmanager
def engine_role(role: str) -> Iterator[None]:
    token = _ROLE.set(role)
    try:
        yield
    finally:
        _ROLE.reset(token)


def inspection_role() -> str | None:
    session = _ACTIVE.get()
    return session.role if session is not None and session.mode == "inspect" else None


def record_attention(network: Any, attention: Any) -> None:
    session = _ACTIVE.get()
    if session is not None and attention is not None:
        session.attentions.setdefault(id(network), []).append(attention)


def _metadata(session: _Session) -> dict[str, Any]:
    return {**session.metadata, "engine_role": _ROLE.get()}


def _dtype(value: str) -> str:
    text = value.upper()
    if "BF16" in text or "BFLOAT16" in text:
        return "bfloat16"
    if "HALF" in text or "FLOAT16" in text:
        return "float16"
    if "INT32" in text:
        return "int32"
    if "FLOAT" in text:
        return "float32"
    raise GraphPatchError(f"unsupported slot tensor dtype {value!r}")


def _output_specs(snapshot: Any, selection: RegionSelection) -> list[dict[str, Any]]:
    tensors = {tensor.id: tensor for tensor in snapshot.tensors}
    inputs = [tensors[tensor_id] for tensor_id in selection.input_tensor_ids]
    boundary = inputs + [
        tensors[tensor_id] for tensor_id in selection.output_tensor_ids
    ]
    for tensor in boundary:
        _dtype(tensor.dtype)
        if len(tensor.shape) > 8:
            raise GraphPatchError(f"slot tensor {tensor.id} exceeds the rank-8 limit")
    output = tensors[selection.output_tensor_ids[0]]
    dims: str | list[int]
    if selection.output_shape_input is not None:
        dims = f"same_as_input_{selection.output_shape_input}"
    else:
        dims = list(output.shape)
    return [{"dims": dims, "dtype": _dtype(output.dtype)}]


def process_network(network: Any) -> None:
    """Capture or patch the current role immediately before TRT serialization."""

    session = _ACTIVE.get()
    role = _ROLE.get()
    if session is None or role != session.role:
        return
    if session.completed:
        raise GraphPatchError(f"engine role {role!r} was built more than once")
    metadata = _metadata(session)
    attentions = session.attentions.get(id(network), ())
    snapshot = snapshot_network(network, metadata=metadata, attentions=attentions)
    if session.mode == "inspect":
        assert session.path is not None
        write_snapshot(snapshot, session.path)
        session.completed = True
        raise GraphInspectionComplete(str(session.path))

    assert session.selection is not None
    selection = session.selection
    specs = _output_specs(snapshot, selection)

    def replacement(live_network: Any, inputs: tuple[Any, ...], _: RegionSelection):
        from .tvm_ffi_plugin import add_tvm_ffi_kernel

        return add_tvm_ffi_kernel(
            live_network,
            kernel_name=selection.kernel_name,
            inputs=list(inputs),
            output_specs=specs,
            workspace_bytes=selection.workspace_bytes,
            extra_args=list(selection.extra_args),
        )

    apply_region(
        network,
        selection,
        replacement,
        metadata=metadata,
        attentions=attentions,
    )
    session.slot = {
        "id": selection.binding_id,
        "abi_sha256": selection.abi_sha256,
    }
    session.completed = True


def kernel_slots_section() -> bytes | None:
    session = _ACTIVE.get()
    if session is None or session.slot is None:
        return None
    return json.dumps(
        {"schema_version": 1, "slots": [session.slot]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
