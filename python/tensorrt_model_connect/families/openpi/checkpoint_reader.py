# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, dependency-light readers for OpenPI parameter trees."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class CheckpointReadError(RuntimeError):
    """Raised when a checkpoint cannot be represented as an array inventory."""


def tensor_sha256(array: np.ndarray) -> str:
    """Hash canonical C-order tensor bytes.

    Shape and dtype are carried separately in manifests.  Keeping the tensor
    digest byte-only makes it directly comparable with exported binary data.
    """

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorRecord:
    name: str
    array: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.array.shape)

    @property
    def dtype(self) -> str:
        return self.array.dtype.name

    @property
    def sha256(self) -> str:
        return tensor_sha256(self.array)

    def manifest_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


def _key_component(value: object) -> str:
    component = str(value)
    if not component or component in {".", ".."} or "/" in component:
        raise CheckpointReadError(f"invalid checkpoint key component: {component!r}")
    return component


def _flatten_tree(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, (*path, _key_component(key)))
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))
            return
        if not path:
            raise CheckpointReadError("checkpoint root must contain named tensors")
        try:
            array = np.asarray(item)
        except (TypeError, ValueError) as exc:
            raise CheckpointReadError(
                f"checkpoint leaf {'/'.join(path)!r} is not array-like"
            ) from exc
        if array.dtype.hasobject:
            raise CheckpointReadError(f"checkpoint leaf {'/'.join(path)!r} has object dtype")
        name = "/".join(path)
        if name in flattened:
            raise CheckpointReadError(f"duplicate flattened checkpoint tensor: {name}")
        flattened[name] = np.ascontiguousarray(array)

    visit(value, prefix)
    return flattened


def _unwrap_params(payload: Any) -> Any:
    if isinstance(payload, Mapping) and set(payload) == {"params"}:
        return payload["params"]
    return payload


def _remove_uniform_value_suffix(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove NNX's leaf ``value`` component only when it is globally present."""

    if arrays and all(name.endswith("/value") for name in arrays):
        normalized: dict[str, np.ndarray] = {}
        for name, array in arrays.items():
            target = name.removesuffix("/value")
            if target in normalized:
                raise CheckpointReadError(
                    f"duplicate tensor after removing NNX value suffix: {target}"
                )
            normalized[target] = array
        return normalized
    return dict(arrays)


class CheckpointReader(Mapping[str, np.ndarray]):
    """Immutable, validated tensor inventory."""

    def __init__(self, arrays: Mapping[str, Any]):
        normalized: dict[str, np.ndarray] = {}
        for raw_name, raw_array in arrays.items():
            name = str(raw_name).strip("/")
            if not name:
                raise CheckpointReadError("checkpoint tensor name cannot be empty")
            if name in normalized:
                raise CheckpointReadError(f"duplicate checkpoint tensor: {name}")
            array = np.asarray(raw_array)
            if array.dtype.hasobject:
                raise CheckpointReadError(f"checkpoint tensor {name!r} has object dtype")
            normalized[name] = np.ascontiguousarray(array)
        if not normalized:
            raise CheckpointReadError("checkpoint contains no tensors")
        self._arrays = dict(sorted(normalized.items()))

    @classmethod
    def from_tree(cls, payload: Any) -> "CheckpointReader":
        arrays = _flatten_tree(_unwrap_params(payload))
        return cls(_remove_uniform_value_suffix(arrays))

    @classmethod
    def from_npz(cls, path: str | Path) -> "CheckpointReader":
        checkpoint = Path(path)
        try:
            with np.load(checkpoint, allow_pickle=False) as archive:
                return cls({name: archive[name] for name in archive.files})
        except (OSError, ValueError) as exc:
            raise CheckpointReadError(f"cannot read NumPy checkpoint {checkpoint}: {exc}") from exc

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def record(self, name: str) -> TensorRecord:
        return TensorRecord(name=name, array=self[name])

    @property
    def identity_sha256(self) -> str:
        """Stable identity over tensor names, shapes, dtypes, and payloads."""

        digest = hashlib.sha256()
        for name in self:
            record = self.record(name)
            fields = (
                name,
                record.dtype,
                ",".join(str(dim) for dim in record.shape),
                record.sha256,
            )
            encoded = "\0".join(fields).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()

    def manifest_inventory(self) -> list[dict[str, object]]:
        return [self.record(name).manifest_entry() for name in self]


def _restore_orbax_payload(path: Path) -> Any:
    """Restore an Orbax PyTree while keeping Orbax and JAX lazy dependencies."""

    try:
        import orbax.checkpoint as ocp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CheckpointReadError(
            "Orbax support is optional; install the OpenPI build-time checkpoint dependencies"
        ) from exc

    try:
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(str(path))
            # Orbax 0.11 returns a TreeMetadata wrapper rather than a Mapping.
            # Its ``tree`` property is the actual checkpoint PyTree.  Looking
            # only for Mapping here silently fell back to an untyped restore,
            # which attempts to recreate the checkpoint's saved device mesh
            # and fails for sharded checkpoints on a CPU-only conversion host.
            metadata_tree = getattr(metadata, "tree", metadata)
            if isinstance(metadata_tree, Mapping) and "params" in metadata_tree:
                # Released OpenPI checkpoints store a {params: ...} PyTree.  A
                # typed restore is needed by current Orbax versions to obtain
                # host NumPy arrays without instantiating the OpenPI model.
                try:
                    import jax  # type: ignore[import-not-found]

                    item = {"params": metadata_tree["params"]}
                    restore_args = jax.tree.map(
                        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item
                    )
                    args = ocp.args.PyTreeRestore(item=item, restore_args=restore_args)
                    return checkpointer.restore(str(path), args)["params"]
                except (AttributeError, ImportError, TypeError):
                    # Compatibility path for older Orbax releases.
                    pass
            return _unwrap_params(checkpointer.restore(str(path)))
    except CheckpointReadError:
        raise
    except Exception as exc:
        raise CheckpointReadError(f"cannot restore Orbax checkpoint {path}: {exc}") from exc


def open_checkpoint(path: str | Path) -> CheckpointReader:
    """Open a synthetic NPZ or an official OpenPI ``params`` checkpoint."""

    checkpoint = Path(path).expanduser()
    if checkpoint.is_file() and checkpoint.suffix == ".npz":
        return CheckpointReader.from_npz(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if checkpoint.is_dir() and (checkpoint / "params").is_dir():
        checkpoint = checkpoint / "params"
    if not checkpoint.is_dir():
        raise CheckpointReadError(
            f"unsupported checkpoint format at {checkpoint}; expected .npz or Orbax directory"
        )
    return CheckpointReader.from_tree(_restore_orbax_payload(checkpoint.resolve()))
