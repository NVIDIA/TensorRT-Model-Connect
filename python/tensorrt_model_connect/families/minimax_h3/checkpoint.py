# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loading for the public Diffusers MiniMax-H3 checkpoint."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import ml_dtypes
import numpy as np


FASTH3_VSA_ADAPTER_SHA256 = (
    "42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470"
)
FASTH3_VSA_ADAPTER_BYTES = 5_339_117_712
FASTH3_VSA_ADAPTER_MODEL_ID = (
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/vsa-datafree"
)
FASTH3_VSA_ADAPTER_SOURCE_REVISION = "317ac01648c4d367ec792e960ece12abe38662b0"
FASTH3_VSA_ADAPTER_BASE_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
FASTH3_VSA_ADAPTER_FINETUNED_REVISION = "b790390377918066c5f5902ec6cc96e21a55926e"
_FASTH3_ADAPTER_METADATA = {
    "format": "fastvideo-lora-v2",
    "base_model": "MiniMaxAI/MiniMax-H3",
    "base_revision": FASTH3_VSA_ADAPTER_BASE_REVISION,
    "finetuned_model": "FastVideo/FastVideo-FastH3-4-step-v1",
    "finetuned_revision": FASTH3_VSA_ADAPTER_FINETUNED_REVISION,
    "rank": "64",
    "low_rank_tensors": "724",
    "diff_tensors": "82",
    "set_weight_tensors": "50",
    "application": (
        "W = W_base + lora_B @ lora_A; then .diff/.diff_b added and "
        ".set_weight assigned"
    ),
}
_FASTH3_ADAPTER_SUFFIXES = (
    (".lora_A.weight", "lora_a", ".weight"),
    (".lora_B.weight", "lora_b", ".weight"),
    (".set_weight", "set_weight", ".weight"),
    (".diff_b", "diff_b", ".bias"),
    (".diff", "diff", ".weight"),
)
_HASH_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class FastH3AdapterIdentity:
    """Auditable identity and exhaustive tensor accounting for FastH3 VSA."""

    sha256: str
    size_bytes: int
    tensor_count: int
    low_rank_tensor_count: int
    diff_tensor_count: int
    set_weight_tensor_count: int
    gate_tensor_count: int
    partition_tensor_counts: dict[str, int]
    metadata: dict[str, str]

    def bundle_metadata(self) -> dict[str, object]:
        """Return path-free metadata safe to persist in a distributable bundle."""

        return {
            "schema_version": 1,
            "adapter_model_id": FASTH3_VSA_ADAPTER_MODEL_ID,
            "adapter_source_revision": FASTH3_VSA_ADAPTER_SOURCE_REVISION,
            "adapter_sha256": self.sha256,
            "adapter_bytes": self.size_bytes,
            "adapter_tensor_count": self.tensor_count,
            "adapter_low_rank_tensor_count": self.low_rank_tensor_count,
            "adapter_diff_tensor_count": self.diff_tensor_count,
            "adapter_set_weight_tensor_count": self.set_weight_tensor_count,
            "adapter_gate_tensor_count": self.gate_tensor_count,
            "adapter_partition_tensor_counts": dict(self.partition_tensor_counts),
            "adapter_base_revision": self.metadata["base_revision"],
            "adapter_finetuned_revision": self.metadata["finetuned_revision"],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fast_h3_adapter_target(name: str) -> tuple[str, str]:
    for suffix, operation, target_suffix in _FASTH3_ADAPTER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + target_suffix, operation
    raise ValueError(f"Unsupported FastH3 adapter tensor: {name!r}")


def validate_fast_h3_adapter(
    adapter_path: str | Path,
    target_partitions: Mapping[str, Iterable[str]],
    *,
    expected_sha256: str = FASTH3_VSA_ADAPTER_SHA256,
    expected_size_bytes: int = FASTH3_VSA_ADAPTER_BYTES,
) -> FastH3AdapterIdentity:
    """Verify the exact published adapter and account for every tensor.

    The adapter is a hybrid payload, not a generic PEFT LoRA.  Every key must
    map to exactly one build component before any multi-gigabyte base tensor is
    materialized.  This prevents silently dropping exact deltas or the 50 VSA
    gate-replacement matrices.
    """

    from safetensors import safe_open

    path = Path(adapter_path).resolve(strict=True)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("FastH3 adapter must be a .safetensors file")
    if not isinstance(expected_size_bytes, int) or expected_size_bytes <= 0:
        raise ValueError("FastH3 expected adapter size must be a positive integer")
    size_bytes = path.stat().st_size
    if size_bytes != expected_size_bytes:
        raise ValueError(
            "FastH3 adapter size mismatch: "
            f"expected={expected_size_bytes}, actual={size_bytes}"
        )
    sha256 = _sha256_file(path)
    if sha256 != expected_sha256:
        raise ValueError(
            "FastH3 adapter SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={sha256}"
        )

    partitions = {name: frozenset(values) for name, values in target_partitions.items()}
    if not partitions or any(not name or not values for name, values in partitions.items()):
        raise ValueError("FastH3 adapter target partitions must be non-empty")
    target_owner: dict[str, str] = {}
    overlaps: dict[str, set[str]] = {}
    for partition, targets in partitions.items():
        for target in targets:
            previous = target_owner.setdefault(target, partition)
            if previous != partition:
                overlaps.setdefault(target, {previous}).add(partition)
    if overlaps:
        raise ValueError(f"FastH3 adapter target partitions overlap: {overlaps}")

    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = dict(reader.metadata() or {})
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in _FASTH3_ADAPTER_METADATA.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"FastH3 adapter metadata mismatch: {mismatches}")
        keys = tuple(reader.keys())
        accounting = {name: 0 for name in partitions}
        operations: dict[str, dict[str, str]] = {}
        unassigned: list[str] = []
        gate_count = 0
        for key in keys:
            target, operation = _fast_h3_adapter_target(key)
            partition = target_owner.get(target)
            if partition is None:
                unassigned.append(key)
                continue
            accounting[partition] += 1
            target_operations = operations.setdefault(target, {})
            if operation in target_operations:
                raise ValueError(
                    f"Duplicate FastH3 {operation} payload for {target!r}: "
                    f"{target_operations[operation]!r}, {key!r}"
                )
            target_operations[operation] = key
            if operation == "set_weight" and target.endswith(
                ".attn.to_gate_compress.weight"
            ):
                gate_count += 1
        if unassigned:
            raise ValueError(
                "FastH3 adapter key accounting is not exhaustive: "
                f"unassigned={sorted(unassigned)}"
            )
        incomplete = {
            target: sorted(set(("lora_a", "lora_b")) ^ (set(parts) & {"lora_a", "lora_b"}))
            for target, parts in operations.items()
            if bool("lora_a" in parts) != bool("lora_b" in parts)
        }
        if incomplete:
            raise ValueError(f"FastH3 adapter has incomplete low-rank pairs: {incomplete}")

    low_rank_count = sum(
        operation in {"lora_a", "lora_b"}
        for parts in operations.values()
        for operation in parts
    )
    diff_count = sum(
        operation in {"diff", "diff_b"}
        for parts in operations.values()
        for operation in parts
    )
    set_count = sum(
        operation == "set_weight"
        for parts in operations.values()
        for operation in parts
    )
    expected_counts = {
        "tensor_count": int(metadata["low_rank_tensors"])
        + int(metadata["diff_tensors"])
        + int(metadata["set_weight_tensors"]),
        "low_rank_tensor_count": int(metadata["low_rank_tensors"]),
        "diff_tensor_count": int(metadata["diff_tensors"]),
        "set_weight_tensor_count": int(metadata["set_weight_tensors"]),
        "gate_tensor_count": 50,
    }
    actual_counts = {
        "tensor_count": len(keys),
        "low_rank_tensor_count": low_rank_count,
        "diff_tensor_count": diff_count,
        "set_weight_tensor_count": set_count,
        "gate_tensor_count": gate_count,
    }
    if actual_counts != expected_counts:
        raise ValueError(
            "FastH3 adapter tensor accounting mismatch: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    return FastH3AdapterIdentity(
        sha256=sha256,
        size_bytes=size_bytes,
        tensor_count=len(keys),
        low_rank_tensor_count=low_rank_count,
        diff_tensor_count=diff_count,
        set_weight_tensor_count=set_count,
        gate_tensor_count=gate_count,
        partition_tensor_counts=accounting,
        metadata=metadata,
    )


def merge_fast_h3_adapter_state(
    state: dict[str, Any],
    adapter_path: str | Path,
    target_keys: Iterable[str],
) -> dict[str, int]:
    """Merge one exhaustively validated adapter partition in FP32.

    Low-rank tensors use ``B @ A`` with implicit ``alpha=rank`` (scale 1),
    exact ``.diff``/``.diff_b`` payloads are additive, and ``.set_weight`` is
    applied last as a replacement.  The returned accounting is suitable for
    comparing against :class:`FastH3AdapterIdentity`.
    """

    from safetensors import safe_open

    allowed = frozenset(target_keys)
    if not allowed:
        raise ValueError("FastH3 adapter merge target set must be non-empty")
    grouped: dict[str, dict[str, str]] = {}
    with safe_open(Path(adapter_path), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            target, operation = _fast_h3_adapter_target(key)
            if target not in allowed:
                continue
            grouped.setdefault(target, {})[operation] = key

        counts = {"low_rank": 0, "diff": 0, "set_weight": 0, "tensors": 0}
        for target, parts in sorted(grouped.items()):
            value = state.get(target)
            has_pair = "lora_a" in parts or "lora_b" in parts
            if has_pair:
                if value is None or set(parts) & {"lora_a", "lora_b"} != {
                    "lora_a",
                    "lora_b",
                }:
                    raise ValueError(f"FastH3 low-rank target is incomplete or absent: {target}")
                a = reader.get_tensor(parts["lora_a"]).float()
                b = reader.get_tensor(parts["lora_b"]).float()
                if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                    raise ValueError(f"FastH3 low-rank factors have invalid shapes: {target}")
                if int(a.shape[0]) != 64:
                    raise ValueError(f"FastH3 adapter rank is not 64: {target}")
                delta = b @ a
                if tuple(delta.shape) != tuple(value.shape):
                    raise ValueError(
                        f"FastH3 low-rank target shape mismatch for {target}: "
                        f"base={tuple(value.shape)}, delta={tuple(delta.shape)}"
                    )
                state[target] = (value.float() + delta).to(value.dtype)
                value = state[target]
                counts["low_rank"] += 1
                counts["tensors"] += 2

            for operation in ("diff", "diff_b"):
                key = parts.get(operation)
                if key is None:
                    continue
                if value is None:
                    raise ValueError(f"FastH3 additive target is absent: {target}")
                delta = reader.get_tensor(key).float()
                if tuple(delta.shape) != tuple(value.shape):
                    raise ValueError(
                        f"FastH3 additive target shape mismatch for {target}: "
                        f"base={tuple(value.shape)}, delta={tuple(delta.shape)}"
                    )
                state[target] = (value.float() + delta).to(value.dtype)
                value = state[target]
                counts["diff"] += 1
                counts["tensors"] += 1

            replacement_key = parts.get("set_weight")
            if replacement_key is not None:
                replacement = reader.get_tensor(replacement_key)
                if value is not None and tuple(replacement.shape) != tuple(value.shape):
                    raise ValueError(
                        f"FastH3 replacement target shape mismatch for {target}: "
                        f"base={tuple(value.shape)}, replacement={tuple(replacement.shape)}"
                    )
                state[target] = replacement
                counts["set_weight"] += 1
                counts["tensors"] += 1
    return counts


class _TensorOwnedArray(np.ndarray):
    """NumPy view that keeps its source checkpoint tensor alive."""

    _tensor_owner: Any

    def __array_finalize__(self, source) -> None:
        if source is not None:
            self._tensor_owner = getattr(source, "_tensor_owner", None)


def load_component_state_dict(component_dir: str | Path) -> dict[str, Any]:
    """Load a safetensors component without materializing duplicate tensors."""

    from safetensors.torch import load_file

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f"Expected one safetensors index in {root}, found {len(indexes)}")
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        paths = [root / name for name in sorted(set(weight_map.values()))]
    else:
        paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors checkpoint found in {root}")

    state: dict[str, Any] = {}
    for path in paths:
        for name, tensor in load_file(path, device="cpu").items():
            if name in state:
                raise ValueError(f"Duplicate MiniMax-H3 tensor {name!r}")
            state[name] = tensor
    return state


def load_selected_component_state_dict(
    component_dir: str | Path, names: Iterable[str]
) -> dict[str, Any]:
    """Load selected tensors from one indexed or unsharded component."""

    from safetensors import safe_open

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    requested = tuple(names)
    state: dict[str, Any] = {}
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f"Selective loading requires one safetensors index in {root}")
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        missing = sorted(set(requested) - set(weight_map))
        if missing:
            raise ValueError(f"MiniMax-H3 checkpoint is missing tensors: {missing}")
        by_file: dict[str, list[str]] = {}
        for name in requested:
            by_file.setdefault(weight_map[name], []).append(name)
        for filename, tensor_names in sorted(by_file.items()):
            with safe_open(root / filename, framework="pt", device="cpu") as reader:
                for name in tensor_names:
                    state[name] = reader.get_tensor(name)
        return state

    paths = sorted(root.glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError(
            f"Selective loading requires one safetensors index or one unsharded file in {root}"
        )
    with safe_open(paths[0], framework="pt", device="cpu") as reader:
        available = set(reader.keys())
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"MiniMax-H3 checkpoint is missing tensors: {missing}")
        for name in requested:
            state[name] = reader.get_tensor(name)
    return state


def validate_component_key_partition(
    component_dir: str | Path, groups: Iterable[Iterable[str]]
) -> None:
    """Require selected groups to partition an indexed component exactly."""

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(f"Partition validation requires one safetensors index in {root}")
    indexed = set(json.loads(indexes[0].read_text())["weight_map"])
    selected: set[str] = set()
    overlap: set[str] = set()
    for group in groups:
        names = set(group)
        overlap.update(selected & names)
        selected.update(names)
    if overlap:
        raise ValueError(f"MiniMax-H3 checkpoint partitions overlap: {sorted(overlap)}")
    missing = sorted(selected - indexed)
    unassigned = sorted(indexed - selected)
    if missing or unassigned:
        raise ValueError(
            "MiniMax-H3 checkpoint partition is not exhaustive: "
            f"missing={missing}, unassigned={unassigned}"
        )


def require_keys(state: dict[str, Any], names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(state))
    if missing:
        raise ValueError(f"MiniMax-H3 checkpoint is missing tensors: {missing}")


def numpy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Expose CPU tensors to NumPy without expanding checkpoint-native BF16.

    NumPy does not natively understand PyTorch BF16, so ``Tensor.numpy()`` is
    not available for those tensors.  Viewing the storage as ``uint16`` first
    and then as :mod:`ml_dtypes` BF16 preserves every checkpoint bit while
    keeping the NumPy array zero-copy.  TensorRT consumes that buffer through
    its explicit BF16 ``Weights`` constructor.
    """

    arrays: dict[str, Any] = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        if str(value.dtype) == "torch.bfloat16":
            # ``Tensor.numpy()`` rejects BF16, so expose its CPU storage as
            # uint16 before reinterpreting the same bits.
            storage_type = ctypes.c_uint16 * value.numel()
            storage = storage_type.from_address(value.data_ptr())
            raw = np.ctypeslib.as_array(storage).reshape(tuple(value.shape)).view(_TensorOwnedArray)
            raw._tensor_owner = value
            arrays[name] = raw.view(ml_dtypes.bfloat16).reshape(tuple(value.shape))
        else:
            arrays[name] = np.asarray(value.numpy())
    return arrays
