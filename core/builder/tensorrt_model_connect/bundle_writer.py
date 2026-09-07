# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal streaming writer for model-family bundles."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator


BUNDLE_MAGIC = b"BUNDLE\x01\x00"
BUNDLE_PROVENANCE_MAGIC = b"PROV\x01\x00\x00\x00"
_FORMAT = 1
_MAX_UINT64 = (1 << 64) - 1
_MAX_HEADER_SIZE = 100 * 1024 * 1024
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def _validate_id(field: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a lowercase identifier containing only "
            "letters, digits, and underscores"
        )
    return value


def _validate_nonempty_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_exact_keys(
    value: object, expected: frozenset[str], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    actual = set(value)
    unsupported = sorted(actual - expected)
    if unsupported:
        raise ValueError(f"{context} contains unsupported field {unsupported[0]!r}")
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{context} missing required field {missing[0]!r}")
    return value


def _require_uint64(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_UINT64
    ):
        raise ValueError(f"{field} must be a non-negative uint64 integer")
    return value


def _validate_bundle_header(
    header: object, *, data_start: int, payload_end: int, path: Path
) -> None:
    parsed = _require_exact_keys(
        header,
        frozenset({"format", "family", "task", "backend", "sections"}),
        context="bundle header",
    )
    if _require_uint64(parsed["format"], field="bundle format") != _FORMAT:
        raise ValueError(f"{path} has an unsupported bundle format")
    for field in ("family", "task", "backend"):
        _validate_nonempty_string(f"bundle header {field}", parsed[field])
    sections = parsed["sections"]
    if not isinstance(sections, dict):
        raise ValueError("bundle header sections must be a JSON object")
    if data_start > payload_end:
        raise ValueError(f"{path} has an invalid section payload range")
    payload_size = payload_end - data_start
    for name, raw_descriptor in sections.items():
        _validate_nonempty_string("bundle section name", name)
        descriptor = _require_exact_keys(
            raw_descriptor,
            frozenset({"offset", "length"}),
            context=f"bundle section {name!r}",
        )
        offset = _require_uint64(
            descriptor["offset"], field=f"bundle section {name!r} offset"
        )
        length = _require_uint64(
            descriptor["length"], field=f"bundle section {name!r} length"
        )
        if offset > payload_size or length > payload_size - offset:
            raise ValueError(f"bundle section {name!r} extends outside {path}")


def read_bundle_provenance(path: str | Path) -> Any:
    """Read the core-owned provenance trailer from a bundle file."""

    bundle_path = Path(path)
    with bundle_path.open("rb") as bundle:
        if bundle.read(len(BUNDLE_MAGIC)) != BUNDLE_MAGIC:
            raise ValueError(f"{bundle_path} is not a TRTMC bundle")
        raw_header_size = bundle.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"{bundle_path} has a truncated header size")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        if header_size > _MAX_HEADER_SIZE:
            raise ValueError(f"{bundle_path} header exceeds the size limit")
        raw_header = bundle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"{bundle_path} has a truncated header")
        header = json.loads(raw_header)
        minimum_start = len(BUNDLE_MAGIC) + 8 + header_size
        bundle.seek(0, os.SEEK_END)
        file_size = bundle.tell()
        footer_size = 8 + len(BUNDLE_PROVENANCE_MAGIC)
        if file_size < minimum_start + footer_size:
            raise ValueError(f"{bundle_path} has no provenance trailer")
        bundle.seek(file_size - len(BUNDLE_PROVENANCE_MAGIC))
        if bundle.read(len(BUNDLE_PROVENANCE_MAGIC)) != BUNDLE_PROVENANCE_MAGIC:
            raise ValueError(f"{bundle_path} has no provenance trailer")
        bundle.seek(file_size - footer_size)
        provenance_size = struct.unpack("<Q", bundle.read(8))[0]
        provenance_start = file_size - footer_size - provenance_size
        if provenance_size > _MAX_HEADER_SIZE or provenance_start < minimum_start:
            raise ValueError(f"{bundle_path} has an invalid provenance trailer")
        _validate_bundle_header(
            header,
            data_start=minimum_start,
            payload_end=provenance_start,
            path=bundle_path,
        )
        bundle.seek(provenance_start)
        raw_provenance = bundle.read(provenance_size)
        if len(raw_provenance) != provenance_size:
            raise ValueError(f"{bundle_path} has a truncated provenance trailer")
        provenance = json.loads(raw_provenance.decode("utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError(f"{bundle_path} provenance must be a JSON object")
        return provenance


class BundleWriter:
    """Stage named sections and atomically publish one bundle."""

    def __init__(self, destination: str | Path) -> None:
        self._destination = Path(destination)
        if not self._destination.parent.is_dir():
            raise FileNotFoundError(
                f"bundle output directory does not exist: {self._destination.parent}"
            )
        self._header: dict[str, Any] | None = None
        self._provenance: bytes | None = None
        self._sections: list[tuple[str, Path]] = []
        self._section_names: set[str] = set()
        self._staging_dir: Path | None = None
        self._open_sections = 0
        self._failed_section = False
        self._finished = False
        self._aborted = False

    def _ensure_writable(self) -> None:
        if self._finished:
            raise RuntimeError("bundle is already finished")
        if self._aborted:
            raise RuntimeError("bundle is aborted")

    def _ensure_staging_dir(self) -> Path:
        if self._staging_dir is None:
            directory = tempfile.mkdtemp(
                prefix=f".{self._destination.name}.sections.",
                dir=self._destination.parent,
            )
            self._staging_dir = Path(directory)
        return self._staging_dir

    def _cleanup_staging(self) -> None:
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

    def set_header(self, *, family: str, task: str, backend: str) -> None:
        """Set the complete shared header exactly once."""

        self._ensure_writable()
        if self._header is not None:
            raise RuntimeError("bundle header is already set")
        self._header = {
            "format": _FORMAT,
            "family": _validate_id("family", family),
            "task": _validate_id("task", task),
            "backend": _validate_id("backend", backend),
        }

    def set_provenance(self, value: Any) -> None:
        """Set the core-owned provenance trailer exactly once."""

        self._ensure_writable()
        if self._provenance is not None:
            raise RuntimeError("bundle provenance is already set")
        if not isinstance(value, dict):
            raise TypeError("bundle provenance must be a JSON object")
        self._provenance = json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(self._provenance) > _MAX_HEADER_SIZE:
            raise ValueError("bundle provenance exceeds the 100 MiB runtime limit")

    @contextmanager
    def open_section(self, name: str) -> Iterator[BinaryIO]:
        """Open one file-backed section for incremental binary writes."""

        self._ensure_writable()
        name = _validate_nonempty_string("section name", name)
        if name in self._section_names:
            raise ValueError(f"duplicate bundle section name: {name!r}")

        section_path = self._ensure_staging_dir() / f"section-{len(self._sections)}"
        self._section_names.add(name)
        self._sections.append((name, section_path))
        self._open_sections += 1
        try:
            with section_path.open("xb") as section:
                yield section
        except BaseException:
            self._failed_section = True
            raise
        finally:
            self._open_sections -= 1

    def add_bytes(self, name: str, data: bytes) -> None:
        """Add a complete in-memory binary section."""

        if not isinstance(data, bytes):
            raise TypeError("section data must be bytes")
        with self.open_section(name) as section:
            section.write(data)

    def add_json(self, name: str, value: Any) -> None:
        """Encode a value as UTF-8 JSON in one section."""

        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.add_bytes(name, data)

    def finish(self) -> None:
        """Write the staged bundle and atomically replace the destination."""

        self._ensure_writable()
        if self._header is None:
            raise RuntimeError("bundle header is not set")
        if self._open_sections:
            raise RuntimeError("cannot finish while a section is open")
        if self._failed_section:
            raise RuntimeError("cannot finish after a section write failed")

        section_table: dict[str, dict[str, int]] = {}
        offset = 0
        for name, path in self._sections:
            length = path.stat().st_size
            if length > _MAX_UINT64 - offset:
                raise OverflowError("bundle section table exceeds uint64 range")
            section_table[name] = {"offset": offset, "length": length}
            offset += length

        header = {**self._header, "sections": section_table}
        header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(header_bytes) > _MAX_HEADER_SIZE:
            raise ValueError("bundle header exceeds the 100 MiB runtime limit")

        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._destination.name}.",
                suffix=".tmp",
                dir=self._destination.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                output.write(BUNDLE_MAGIC)
                output.write(struct.pack("<Q", len(header_bytes)))
                output.write(header_bytes)
                for _, section_path in self._sections:
                    with section_path.open("rb") as section:
                        shutil.copyfileobj(section, output)
                if self._provenance is not None:
                    output.write(self._provenance)
                    output.write(struct.pack("<Q", len(self._provenance)))
                    output.write(BUNDLE_PROVENANCE_MAGIC)
            os.replace(temporary_path, self._destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        self._finished = True
        self._cleanup_staging()

    def abort(self) -> None:
        """Discard staged data without changing the destination."""

        if self._finished or self._aborted:
            return
        self._aborted = True
        self._cleanup_staging()
