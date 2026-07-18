# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write .trtfb bundle files — 1:1 compatible with C++ ReadBundleFile().

Format:
  Bytes 0-7:   Magic "TRTFB\\x00\\x01\\x00"
  Bytes 8-15:  uint64_t json_header_length (LE)
  Bytes 16..N: JSON metadata header (UTF-8)
  Bytes N..EOF: Binary sections
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
_MAX_BUNDLE_HEADER_SIZE = 100 * 1024 * 1024


@dataclass
class BundleInfo:
    model_id: str = ""
    model_type: str = ""
    family: str = ""
    trt_version: str = ""
    trt_abi: str = ""
    gpu_name: str = ""
    created_at: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    max_cache_length: int = 32
    runtime_strategy: str = ""
    precision: str = "fp32"
    quantization: str = "none"
    tokenizer_add_special_tokens: bool = False
    io_map: dict | None = None  # tensor name mapping; None = TRT API defaults
    # Namespaced defaults produced at build time. When non-empty, serialized
    # into the header as `defaults: {namespace: {field: value, ...}}` and
    # read back at runtime as the BUNDLE_DEFAULT layer — the lowest-priority
    # input to the config registry merge. None/empty → no block emitted, so
    # old readers continue to work untouched.
    defaults: dict | None = None
    # Per-component batch-size envelope for diffusion bundles. Shape:
    # `{"dit": N, "text_encoder": N, "vae": N}`. None → field is omitted from
    # the JSON header so older runtimes still load the bundle and treat the
    # engine as B=1. See design doc Decision C.
    max_batch_size: dict[str, int] | None = None


@dataclass
class BundleSection:
    name: str
    data: bytes


@dataclass(frozen=True)
class _FileBundleSection:
    name: str
    source_path: Path
    expected_sha256: str | None


def _bundle_section_from_file(
    name: str,
    source_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> BundleSection:
    """Create a private file-backed section for atomic streaming writes."""

    return cast(
        BundleSection,
        _FileBundleSection(
            name=name,
            source_path=Path(source_path),
            expected_sha256=expected_sha256,
        ),
    )


def _section_size(section: BundleSection) -> int:
    if not isinstance(section, _FileBundleSection):
        return len(section.data)
    try:
        source_stat = section.source_path.lstat()
    except OSError as exc:
        raise ValueError(
            f"Bundle section {section.name!r} source is unavailable: {section.source_path}: {exc}"
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(
            f"Bundle section {section.name!r} source is not a regular file: {section.source_path}"
        )
    if section.expected_sha256 is not None and (
        len(section.expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in section.expected_sha256)
    ):
        raise ValueError(f"Bundle section {section.name!r} has invalid expected_sha256")
    return source_stat.st_size


def _write_section(output, section: BundleSection, expected_size: int) -> None:
    if not isinstance(section, _FileBundleSection):
        output.write(section.data)
        return

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(section.source_path, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_size != expected_size:
            raise RuntimeError(
                f"Bundle section source changed before reading: {section.source_path}"
            )
        source = os.fdopen(descriptor, "rb")
        descriptor = None
    except OSError as exc:
        raise RuntimeError(
            "Unable to open bundle section source without following links: "
            f"{section.source_path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    remaining = expected_size
    digest = hashlib.sha256() if section.expected_sha256 is not None else None
    with source:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    f"Bundle section source changed while reading: {section.source_path}"
                )
            output.write(chunk)
            if digest is not None:
                digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise RuntimeError(
                f"Bundle section source changed while reading: {section.source_path}"
            )
    if digest is not None and digest.hexdigest() != section.expected_sha256:
        raise RuntimeError(f"Bundle section source changed after validation: {section.source_path}")


def _open_atomic_bundle_output(destination: Path):
    """Create a same-directory temporary with normal file-create permissions."""

    try:
        destination_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        destination_mode = None

    for _attempt in range(100):
        temporary_path = destination.parent / (
            f".{destination.name}.tmp.{secrets.token_hex(8)}"
        )
        try:
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            continue
        try:
            if destination_mode is not None:
                os.fchmod(descriptor, destination_mode)
            return os.fdopen(descriptor, "wb"), temporary_path
        except Exception:
            os.close(descriptor)
            temporary_path.unlink(missing_ok=True)
            raise
    raise FileExistsError(f"Unable to allocate temporary bundle beside {destination}")


def _write_file_backed_bundle(
    path: str | Path,
    info: BundleInfo,
    sections: list[BundleSection],
) -> None:
    """Atomically stream a bundle containing private file-backed sections."""
    # Build section offset/size list for JSON header
    section_meta: list[dict[str, Any]] = []
    section_sizes: list[int] = []
    section_names: set[str] = set()
    offset = 0
    for s in sections:
        if not s.name or s.name in section_names:
            raise ValueError(f"Invalid or duplicate bundle section name: {s.name!r}")
        section_names.add(s.name)
        size = _section_size(s)
        section_sizes.append(size)
        section_meta.append(
            {
                "name": s.name,
                "offset": offset,
                "size": size,
            }
        )
        offset += size

    # Build JSON header
    header = {
        "model_id": info.model_id,
        "model_type": info.model_type,
        "family": info.family,
        "trt_version": info.trt_version,
        "trt_abi": info.trt_abi,
        "gpu_name": info.gpu_name,
        "created_at": info.created_at,
        "vocab_size": info.vocab_size,
        "hidden_size": info.hidden_size,
        "num_layers": info.num_layers,
        "num_attention_heads": info.num_attention_heads,
        "num_key_value_heads": info.num_key_value_heads,
        "max_cache_length": info.max_cache_length,
        **({"runtime_strategy": info.runtime_strategy} if info.runtime_strategy else {}),
        "precision": info.precision,
        **({"quantization": info.quantization} if info.quantization != "none" else {}),
        "tokenizer_add_special_tokens": int(info.tokenizer_add_special_tokens),
        **({"io_map": info.io_map} if info.io_map else {}),
        **({"defaults": info.defaults} if info.defaults else {}),
        **({"max_batch_size": dict(info.max_batch_size)} if info.max_batch_size else {}),
        "sections": {s["name"]: {"offset": s["offset"], "size": s["size"]} for s in section_meta},
    }
    header_json = json.dumps(header, indent=2).encode("utf-8")
    if len(header_json) > _MAX_BUNDLE_HEADER_SIZE:
        raise ValueError(
            f"Bundle JSON header exceeds the 100 MiB runtime limit: {len(header_json)} bytes"
        )

    destination = Path(path)
    temporary_path: Path | None = None
    try:
        output, temporary_path = _open_atomic_bundle_output(destination)
        with output:
            output.write(BUNDLE_MAGIC)
            output.write(struct.pack("<Q", len(header_json)))
            output.write(header_json)
            for section, size in zip(sections, section_sizes, strict=True):
                _write_section(output, section, size)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_bundle(
    path: str | Path,
    info: BundleInfo,
    sections: list[BundleSection],
) -> None:
    """Write a .trtfb bundle file."""
    if any(isinstance(section, _FileBundleSection) for section in sections):
        _write_file_backed_bundle(path, info, sections)
        return

    # Build section offset/size list for JSON header
    section_meta: list[dict[str, Any]] = []
    offset = 0
    for s in sections:
        section_meta.append({
            "name": s.name,
            "offset": offset,
            "size": len(s.data),
        })
        offset += len(s.data)

    # Build JSON header
    header = {
        "model_id": info.model_id,
        "model_type": info.model_type,
        "family": info.family,
        "trt_version": info.trt_version,
        "trt_abi": info.trt_abi,
        "gpu_name": info.gpu_name,
        "created_at": info.created_at,
        "vocab_size": info.vocab_size,
        "hidden_size": info.hidden_size,
        "num_layers": info.num_layers,
        "num_attention_heads": info.num_attention_heads,
        "num_key_value_heads": info.num_key_value_heads,
        "max_cache_length": info.max_cache_length,
        **({"runtime_strategy": info.runtime_strategy}
           if info.runtime_strategy else {}),
        "precision": info.precision,
        **({"quantization": info.quantization}
           if info.quantization != "none" else {}),
        "tokenizer_add_special_tokens": int(info.tokenizer_add_special_tokens),
        **({"io_map": info.io_map} if info.io_map else {}),
        **({"defaults": info.defaults} if info.defaults else {}),
        **({"max_batch_size": dict(info.max_batch_size)}
           if info.max_batch_size else {}),
        "sections": {
            s["name"]: {"offset": s["offset"], "size": s["size"]}
            for s in section_meta
        },
    }
    header_json = json.dumps(header, indent=2).encode("utf-8")

    with open(path, "wb") as f:
        f.write(BUNDLE_MAGIC)
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for s in sections:
            f.write(s.data)
