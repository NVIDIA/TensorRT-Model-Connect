# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic DINOv3 weighted k-NN task helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

K_VALUES = (10, 20, 100, 200)
TEMPERATURE = 0.07


def load_image_manifest(
    path: str | Path,
) -> tuple[list[Path], np.ndarray, tuple[str, ...]]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    class_names = payload.get("class_names")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{manifest_path}: samples must be a non-empty list")
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) and name for name in class_names)
    ):
        raise ValueError(f"{manifest_path}: class_names must be non-empty strings")
    images: list[Path] = []
    labels: list[int] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"{manifest_path}: sample {index} must be an object")
        image = sample.get("image")
        label = sample.get("label")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{manifest_path}: sample {index} has no image")
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"{manifest_path}: sample {index} has invalid label")
        if label < 0 or label >= len(class_names):
            raise ValueError(f"{manifest_path}: sample {index} label is out of range")
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path}: image does not exist: {image_path}"
            )
        images.append(image_path.resolve())
        labels.append(label)
    return images, np.asarray(labels, dtype=np.int64), tuple(class_names)


def normalize_rows(features: Any) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            f"features must be a non-empty rank-2 array, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("features contain a zero-norm row")
    return values / norms


def weighted_knn_predictions(
    bank_features: Any,
    bank_labels: Any,
    query_features: Any,
    *,
    num_classes: int,
    ks: Sequence[int] = K_VALUES,
    temperature: float = TEMPERATURE,
) -> dict[str, list[int]]:
    bank = normalize_rows(bank_features)
    queries = normalize_rows(query_features)
    labels = np.asarray(bank_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != bank.shape[0]:
        raise ValueError("bank labels must match the bank feature rows")
    if num_classes <= 0 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("bank labels are outside the configured class range")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    resolved_ks = tuple(sorted(set(int(k) for k in ks)))
    if not resolved_ks or resolved_ks[0] <= 0 or resolved_ks[-1] > len(bank):
        raise ValueError("k values must be positive and no larger than the bank")

    similarities = queries @ bank.T
    top_indices = np.argpartition(-similarities, resolved_ks[-1] - 1, axis=1)[
        :, : resolved_ks[-1]
    ]
    top_values = np.take_along_axis(similarities, top_indices, axis=1)
    order = np.argsort(-top_values, axis=1, kind="stable")
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_values = np.take_along_axis(top_values, order, axis=1)
    neighbor_labels = labels[top_indices]

    scaled = top_values / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=1, keepdims=True)
    predictions: dict[str, list[int]] = {}
    rows = np.arange(len(queries))
    for k in resolved_ks:
        votes = np.zeros((len(queries), num_classes), dtype=np.float64)
        np.add.at(
            votes,
            (np.repeat(rows, k), neighbor_labels[:, :k].reshape(-1)),
            weights[:, :k].reshape(-1),
        )
        predictions[str(k)] = np.argmax(votes, axis=1).astype(int).tolist()
    return predictions


def tensor_payload(features: Any) -> dict[str, Any]:
    values = np.ascontiguousarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("pooler features must be a finite rank-2 array")
    return {"shape": list(values.shape), "data": values.reshape(-1).tolist()}
