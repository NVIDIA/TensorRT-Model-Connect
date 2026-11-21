# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File references carried inside JSON operation requests.

Operation requests name inputs by path (``image_path``, ``audio_path``,
``frame_paths`` ...). Over HTTP a path is replaced by one of two references::

    {"$file": {"suffix": ".png", "b64": "<base64 bytes>"}}      the bytes, inline
    {"$file": {"suffix": ".png", "sha256": "<64 hex digits>"}}  the bytes, kept once in a media store

``inline_files`` builds the first form from a local request and ``externalize_files`` the second: every
image, clip and frame is written once to a content-addressed media store (``<root>/<aa>/<digest><suffix>``)
that all models, sides and runs share, and requests, AIPerf inputs and raw exports carry only the digest.
``materialize_files`` writes either form back to per-request files on the server; the server reads a
reference only from the media store it was started with (a client never names a path) and checks the
bytes against the digest.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

FILE_KEY = "$file"
MAX_SUFFIX_LENGTH = 16
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _is_path_field(name: str) -> bool:
    return name.endswith("_path") or name.endswith("_paths")


def _suffix_of(path: Path) -> str:
    return "".join(path.suffixes)[-MAX_SUFFIX_LENGTH:]


def inline_files(value: Any, *, field: str = "") -> Any:
    """Replace every existing path under a ``*_path``/``*_paths`` field by inline bytes."""
    if isinstance(value, dict):
        return {key: inline_files(item, field=key) for key, item in value.items()}
    if isinstance(value, list):
        return [inline_files(item, field=field) for item in value]
    if isinstance(value, str) and _is_path_field(field) and Path(value).is_file():
        path = Path(value)
        return {FILE_KEY: {"suffix": _suffix_of(path), "b64": base64.b64encode(path.read_bytes()).decode("ascii")}}
    return value


def server_path_fields(value: Any, *, field: str = "") -> list[str]:
    """``*_path``/``*_paths`` fields holding plain strings: server-side paths that a client must not
    make the server read (clients send files as ``$file`` references)."""
    if isinstance(value, dict):
        return [name for key, item in value.items() for name in server_path_fields(item, field=key)]
    if isinstance(value, list):
        return [name for item in value for name in server_path_fields(item, field=field)]
    return [field] if isinstance(value, str) and _is_path_field(field) else []


def _body(reference: dict[str, Any]) -> dict[str, Any]:
    """The validated content of a ``$file`` reference."""
    if set(reference) != {FILE_KEY}:
        raise ValueError(f"{FILE_KEY} references cannot carry sibling fields")
    body = reference[FILE_KEY]
    if not isinstance(body, dict):
        raise ValueError(f"{FILE_KEY} must hold an object")
    suffix = body.get("suffix", "")
    if not isinstance(suffix, str) or len(suffix) > MAX_SUFFIX_LENGTH or "/" in suffix or ".." in suffix:
        raise ValueError(f"{FILE_KEY} suffix must be a short file extension")
    if ("b64" in body) == ("sha256" in body):
        raise ValueError(f"{FILE_KEY} requires exactly one of a base64 'b64' string and a 'sha256' digest")
    if "sha256" in body and not (isinstance(body["sha256"], str) and _DIGEST.fullmatch(body["sha256"])):
        raise ValueError(f"{FILE_KEY} sha256 must be 64 lowercase hex digits")
    return body


def media_path(root: Path, digest: str, suffix: str) -> Path:
    return root / digest[:2] / f"{digest}{suffix}"


def store_media(root: Path, data: bytes, suffix: str) -> dict[str, Any]:
    """Keep ``data`` once in the media store ``root`` and return the reference to it. The copy is named by
    its digest, written atomically (parallel runs may store the same bytes) and read-only."""
    digest = hashlib.sha256(data).hexdigest()
    path = media_path(root, digest, suffix)
    if not path.is_file() or path.stat().st_size != len(data):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as temporary:
                temporary.write(data)
            os.chmod(name, 0o444)
            os.replace(name, path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
    return {FILE_KEY: {"suffix": suffix, "sha256": digest}}


def externalize_files(value: Any, root: Path | None, *, field: str = "") -> Any:
    """Replace the media of a request by media-store references: existing paths under ``*_path``/``*_paths``
    fields and inline ``$file`` bytes are stored once in ``root``; references stay as they are. A request
    without media needs no store."""
    if isinstance(value, dict):
        if FILE_KEY in value:
            body = _body(value)
            if "sha256" in body:
                return value
            return _store(root, _decode(body), body.get("suffix", ""))
        return {key: externalize_files(item, root, field=key) for key, item in value.items()}
    if isinstance(value, list):
        return [externalize_files(item, root, field=field) for item in value]
    if isinstance(value, str) and _is_path_field(field) and Path(value).is_file():
        return _store(root, Path(value).read_bytes(), _suffix_of(Path(value)))
    return value


def _store(root: Path | None, data: bytes, suffix: str) -> dict[str, Any]:
    if root is None:
        raise ValueError("a request with media needs a media store (none is configured)")
    return store_media(root, data, suffix)


def materialize_files(value: Any, directory: Path, media_root: Path | None = None) -> Any:
    """Write file references below ``directory`` and substitute their paths. A ``sha256`` reference is read
    from ``media_root`` and verified; the backend gets its own copy, so a store file is never modified."""
    counter = iter(range(1_000_000))

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            if FILE_KEY in item:
                return _write(item, directory, next(counter), media_root)
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)


def _decode(body: dict[str, Any]) -> bytes:
    if not isinstance(body["b64"], str):
        raise ValueError(f"{FILE_KEY} requires a base64 'b64' string")
    try:
        return base64.b64decode(body["b64"], validate=True)
    except binascii.Error as error:
        raise ValueError(f"{FILE_KEY} contains invalid base64: {error}") from error


def _stored(body: dict[str, Any], media_root: Path | None) -> bytes:
    if media_root is None:
        raise ValueError(f"{FILE_KEY} sha256 references need the server's media store (--media-root)")
    digest = body["sha256"]
    try:
        data = media_path(media_root, digest, body.get("suffix", "")).read_bytes()
    except FileNotFoundError:
        raise ValueError(f"media {digest} is not in the server's media store") from None
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"the media store's copy of {digest} does not match its digest")
    return data


def _write(reference: dict[str, Any], directory: Path, index: int, media_root: Path | None) -> str:
    body = _body(reference)
    data = _stored(body, media_root) if "sha256" in body else _decode(body)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"input-{index:04d}{body.get('suffix', '')}"
    path.write_bytes(data)
    return str(path)
