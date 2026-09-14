# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact NumPy access to a YOLOv5 `.pt` checkpoint.

YOLOv5 predates the `ultralytics` package: its archives pickle classes from the
standalone yolov5 repository's `models` package, which is not a dependency
here. Rather than take that repository on, the classes are stood in for while
the archive is read. Nothing in a stand-in runs - unpickling a module only
restores attributes - so the tensors, the class names and the anchor boxes come
out unchanged, and the topology is taken from the stage table in `model.py`
instead of from the restored objects.

The stored tensors are half precision; they are widened to float32 here so the
folding arithmetic downstream keeps its accuracy.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# The modules a YOLOv5 archive names. Anything it reaches for inside these is
# answered with a placeholder class.
_PICKLED_MODULES = (
    "models",
    "models.yolo",
    "models.common",
    "models.experimental",
    "utils",
    "utils.general",
)


class _Placeholder:
    """Stands in for a class the archive pickles that is not available here."""

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {})


def _placeholder_for(attribute: str) -> type:
    # Dunder lookups have to keep failing: `inspect` walks `__file__` on every
    # module it is handed, and answering that with a class breaks it.
    if attribute.startswith("__") and attribute.endswith("__"):
        raise AttributeError(attribute)
    return type(attribute, (_Placeholder,), {})


def _install_placeholders() -> None:
    for name in _PICKLED_MODULES:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__getattr__ = _placeholder_for  # type: ignore[attr-defined]
        sys.modules[name] = module


def _collect(node: Any, prefix: str, into: dict[str, Any]) -> None:
    """Walk a restored module tree the way `state_dict` would."""
    data = node.__dict__
    for store in ("_parameters", "_buffers"):
        for name, tensor in data.get(store, {}).items():
            if tensor is not None:
                into[f"{prefix}{name}"] = tensor
    for name, child in data.get("_modules", {}).items():
        if child is not None:
            _collect(child, f"{prefix}{name}.", into)


@dataclass(frozen=True)
class Checkpoint:
    tensors: dict[str, np.ndarray]
    class_names: tuple[str, ...]
    image_size: int
    strides: tuple[int, ...]

    # A YOLOv5 release ships every width in one repository, and a build request
    # has no field naming which archive to use, so the family builds the width
    # it is named for. Reading the other widths needs a way for a manifest to
    # select a file, which does not exist today.
    ARCHIVE = "yolov5n.pt"

    @classmethod
    def open(cls, model_dir: Path) -> "Checkpoint":
        path = model_dir / cls.ARCHIVE
        if not path.is_file():
            raise FileNotFoundError(f"YOLOv5 model directory has no {cls.ARCHIVE}: {model_dir}")
        return cls.from_archive(path)

    @classmethod
    def from_archive(cls, path: Path) -> "Checkpoint":
        import torch

        _install_placeholders()
        # The archive pickles its own model class, so it cannot be read with
        # weights_only. The manifest pins the revision it comes from.
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        model = blob.get("model") if isinstance(blob, dict) else None
        if model is None:
            raise ValueError(f"YOLOv5 archive has no model entry: {path}")

        restored: dict[str, Any] = {}
        _collect(model, "", restored)
        tensors = {
            key: value.detach().to(torch.float32).numpy()
            for key, value in restored.items()
            if not key.endswith("num_batches_tracked")
        }
        if not tensors:
            raise ValueError(f"YOLOv5 archive has no weights: {path}")

        names = model.__dict__.get("names")
        if isinstance(names, dict) and names:
            ordered = tuple(str(names[index]) for index in sorted(names))
        elif isinstance(names, (list, tuple)) and names:
            ordered = tuple(str(name) for name in names)
        else:
            raise ValueError(f"YOLOv5 archive has no class names: {path}")

        stride = restored.get("stride")
        if stride is None:
            stride = model.__dict__.get("stride")
        if stride is None:
            raise ValueError(f"YOLOv5 archive records no detection strides: {path}")
        strides = tuple(int(value) for value in stride.detach().reshape(-1).tolist())
        if not strides or any(value <= 0 for value in strides):
            raise ValueError(f"YOLOv5 archive has a non-positive stride: {path}")

        # A YOLOv5 archive does not record the size it was trained at; every
        # published release uses 640.
        return cls(tensors, ordered, 640, strides)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.tensors)

    def exists(self, name: str) -> bool:
        return name in self.tensors

    def tensor(self, name: str) -> np.ndarray:
        value = self.tensors.get(name)
        if value is None:
            raise KeyError(f"YOLOv5 checkpoint tensor not found: {name}")
        return value

    def scalar(self, name: str) -> Any:
        return self.tensor(name)
