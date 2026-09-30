# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline file references carried inside JSON operation requests.

Operation requests name inputs by path (``image_path``, ``audio_path``,
``frame_paths`` ...). Over HTTP a path is replaced by an inline reference::

    {"$file": {"suffix": ".png", "b64": "<base64 bytes>"}}

``inline_files`` builds such payloads from a local request and
``materialize_files`` writes them back to per-request files on the server.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

FILE_KEY = "$file"
MAX_SUFFIX_LENGTH = 16


def _is_path_field(name: str) -> bool:
    return name.endswith("_path") or name.endswith("_paths")


def inline_files(value: Any, *, field: str = "") -> Any:
    """Replace every existing path under a ``*_path``/``*_paths`` field by inline bytes."""
    if isinstance(value, dict):
        return {key: inline_files(item, field=key) for key, item in value.items()}
    if isinstance(value, list):
        return [inline_files(item, field=field) for item in value]
    if isinstance(value, str) and _is_path_field(field) and Path(value).is_file():
        path = Path(value)
        return {FILE_KEY: {"suffix": "".join(path.suffixes)[-MAX_SUFFIX_LENGTH:],
                           "b64": base64.b64encode(path.read_bytes()).decode("ascii")}}
    return value


def server_path_fields(value: Any, *, field: str = "") -> list[str]:
    """``*_path``/``*_paths`` fields holding plain strings: server-side paths that a client must not
    make the server read (clients send files inline as ``$file``)."""
    if isinstance(value, dict):
        return [name for key, item in value.items() for name in server_path_fields(item, field=key)]
    if isinstance(value, list):
        return [name for item in value for name in server_path_fields(item, field=field)]
    return [field] if isinstance(value, str) and _is_path_field(field) else []


def materialize_files(value: Any, directory: Path) -> Any:
    """Write inline references below ``directory`` and substitute their paths."""
    counter = iter(range(1_000_000))

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            if FILE_KEY in item:
                return _write(item, directory, next(counter))
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)


def _write(reference: dict[str, Any], directory: Path, index: int) -> str:
    if set(reference) != {FILE_KEY}:
        raise ValueError(f"{FILE_KEY} references cannot carry sibling fields")
    body = reference[FILE_KEY]
    if not isinstance(body, dict) or not isinstance(body.get("b64"), str):
        raise ValueError(f"{FILE_KEY} requires a base64 'b64' string")
    suffix = body.get("suffix", "")
    if not isinstance(suffix, str) or len(suffix) > MAX_SUFFIX_LENGTH or "/" in suffix or ".." in suffix:
        raise ValueError(f"{FILE_KEY} suffix must be a short file extension")
    try:
        data = base64.b64decode(body["b64"], validate=True)
    except binascii.Error as error:
        raise ValueError(f"{FILE_KEY} contains invalid base64: {error}") from error
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"input-{index:04d}{suffix}"
    path.write_bytes(data)
    return str(path)
