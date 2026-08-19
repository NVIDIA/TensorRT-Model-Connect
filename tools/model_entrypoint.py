# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an entrypoint declared by one model owner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"


def load_model_entrypoint(
    family: str,
    field: str,
) -> Callable[..., Any] | None:
    """Return one declared owner callable, or ``None`` for the generic path.

    A supplied family must resolve to exactly one owner. Declared paths and
    symbols fail closed; only an owner that omits the field uses the generic
    shared implementation.
    """

    owner_id = str(family or "").strip()
    if not owner_id:
        return None
    if owner_id in {".", ".."} or Path(owner_id).name != owner_id:
        raise ValueError(f"invalid model owner id {owner_id!r}")

    owner = MODELS_ROOT / owner_id
    descriptor = owner / "MODEL.toml"
    if not descriptor.is_file():
        raise ValueError(f"unknown model owner {owner_id!r}")
    with descriptor.open("rb") as stream:
        metadata = tomllib.load(stream)
    if metadata.get("id") != owner_id:
        raise ValueError(
            f"model owner descriptor id must match its folder: {descriptor}"
        )

    declaration = metadata.get(field)
    if declaration is None:
        return None
    if not isinstance(declaration, str):
        raise ValueError(f"{descriptor}: {field} must be a string")
    relative_path, separator, symbol = declaration.partition("|")
    if not separator or not relative_path or not symbol:
        raise ValueError(
            f"{descriptor}: {field} must be 'relative/path.py|callable'"
        )

    if Path(relative_path).is_absolute():
        raise ValueError(f"{descriptor}: {field} must be owner-relative")
    resolved_owner = owner.resolve()
    entrypoint = (owner / relative_path).resolve()
    try:
        entrypoint.relative_to(resolved_owner)
    except ValueError as exc:
        raise ValueError(f"{descriptor}: {field} escapes its model owner") from exc
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise FileNotFoundError(
            f"{descriptor}: {field} entrypoint is not a regular file: {entrypoint}"
        )

    module_name = f"_trtmc_{owner_id}_{field}_{entrypoint.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load model entrypoint {entrypoint}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    callable_object = getattr(module, symbol, None)
    if not callable(callable_object):
        raise TypeError(
            f"{descriptor}: {field} symbol {symbol!r} is not callable"
        )
    return callable_object
