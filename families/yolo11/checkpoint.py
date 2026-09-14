# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact NumPy access to an Ultralytics YOLO11 `.pt` checkpoint.

Ultralytics publishes YOLO11 as a pickled torch archive rather than
safetensors, so this reader unpickles it once and hands back plain arrays. The
stored tensors are half precision; they are widened to float32 here so the
folding arithmetic downstream keeps its accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Checkpoint:
    tensors: dict[str, np.ndarray]
    class_names: tuple[str, ...]
    image_size: int

    # An Ultralytics release ships every width in one repository, and a build
    # request has no field naming which archive to use, so the family builds
    # the width it is named for. Reading the other widths needs a way for a
    # manifest to select a file, which does not exist today.
    ARCHIVE = "yolo11n.pt"

    @classmethod
    def open(cls, model_dir: Path) -> "Checkpoint":
        path = model_dir / cls.ARCHIVE
        if not path.is_file():
            raise FileNotFoundError(f"YOLO11 model directory has no {cls.ARCHIVE}: {model_dir}")
        return cls.from_archive(path)

    @classmethod
    def from_archive(cls, path: Path) -> "Checkpoint":
        import torch

        # An Ultralytics archive pickles its own model class, so it cannot be
        # read with weights_only. The manifest pins the revision it comes from.
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        model = blob.get("model") if isinstance(blob, dict) else None
        if model is None:
            raise ValueError(f"YOLO11 archive has no model entry: {path}")
        names = getattr(model, "names", None)
        if not isinstance(names, dict) or not names:
            raise ValueError(f"YOLO11 archive has no class names: {path}")
        state = model.state_dict()
        tensors = {
            str(key): value.detach().to(torch.float32).numpy()
            for key, value in state.items()
            if not str(key).endswith("num_batches_tracked")
        }
        if not tensors:
            raise ValueError(f"YOLO11 archive has no weights: {path}")
        ordered = tuple(str(names[index]) for index in sorted(names))
        # The archive records the size it was trained at; every published
        # YOLO11 release uses 640.
        arguments = blob.get("train_args") if isinstance(blob, dict) else None
        size = 640
        if isinstance(arguments, dict):
            value = arguments.get("imgsz")
            if isinstance(value, int) and value > 0:
                size = int(value)
        return cls(tensors, ordered, size)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.tensors)

    def exists(self, name: str) -> bool:
        return name in self.tensors

    def tensor(self, name: str) -> np.ndarray:
        value = self.tensors.get(name)
        if value is None:
            raise KeyError(f"YOLO11 checkpoint tensor not found: {name}")
        return value

    def scalar(self, name: str) -> Any:
        return self.tensor(name)
