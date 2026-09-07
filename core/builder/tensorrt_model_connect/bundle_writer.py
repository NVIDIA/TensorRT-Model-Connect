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


class BundleWriter:
    """Stage named sections and atomically publish one bundle."""

    def __init__(self, destination: str | Path) -> None:
        self._destination = Path(destination)
        if not self._destination.parent.is_dir():
            raise FileNotFoundError(
                f"bundle output directory does not exist: {self._destination.parent}"
            )
        self._header: dict[str, Any] | None = None
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


def _detect_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate section name: {key!r}")
        seen[key] = value
    return seen


class BundleReader:
    """Read named sections from a bundle produced by BundleWriter."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        with self._path.open("rb") as f:
            magic = f.read(len(BUNDLE_MAGIC))
            if magic != BUNDLE_MAGIC:
                raise ValueError(
                    f"Invalid bundle magic signature: expected {BUNDLE_MAGIC!r}, got {magic!r}"
                )
            (header_size,) = struct.unpack("<Q", f.read(8))
            if header_size > _MAX_HEADER_SIZE:
                raise ValueError("bundle header exceeds the 100 MiB runtime limit")
            header_bytes = f.read(header_size)
            self._data_start = len(BUNDLE_MAGIC) + 8 + header_size
            file_size = self._path.stat().st_size
            self._data_size = file_size - self._data_start

        try:
            header: dict[str, Any] = json.loads(
                header_bytes, object_pairs_hook=_detect_duplicate_keys
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"bundle header is not valid JSON: {exc}") from exc

        if "sections" not in header:
            raise ValueError("bundle header missing 'sections' key")
        raw_sections = header["sections"]
        if not isinstance(raw_sections, dict):
            raise ValueError("bundle header 'sections' must be a JSON object")

        sections: dict[str, tuple[int, int]] = {}
        for name, entry in raw_sections.items():
            offset = entry.get("offset")
            length = entry.get("length")
            if not isinstance(offset, int) or not isinstance(length, int):
                raise ValueError(
                    f"section {name!r}: offset and length must be integers"
                )
            if offset < 0 or length < 0:
                raise ValueError(
                    f"section {name!r}: offset and length must be non-negative"
                )
            if offset + length > self._data_size:
                raise ValueError(
                    f"section {name!r}: goes out-of-file range "
                    f"(offset={offset}, length={length}, data_size={self._data_size})"
                )
            sections[name] = (offset, length)

        # Overlap check: sort by offset and verify no two sections overlap.
        sorted_sections = sorted(sections.items(), key=lambda kv: kv[1][0])
        for i in range(len(sorted_sections) - 1):
            name_a, (off_a, len_a) = sorted_sections[i]
            name_b, (off_b, _) = sorted_sections[i + 1]
            if off_a + len_a > off_b:
                raise ValueError(
                    f"Sections overlap: {name_a!r} ends at {off_a + len_a}, "
                    f"but {name_b!r} starts at {off_b}"
                )

        self._sections = sections

    def read_section(self, name: str) -> bytes:
        """Read and return the raw bytes of a named section."""

        if name not in self._sections:
            raise KeyError(f"bundle has no section named {name!r}")
        offset, length = self._sections[name]
        with self._path.open("rb") as f:
            f.seek(self._data_start + offset)
            return f.read(length)
