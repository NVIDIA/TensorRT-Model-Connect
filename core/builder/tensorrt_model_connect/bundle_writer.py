# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal streaming writer for model-family bundles."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from . import provenance


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


def _write_all(output: BinaryIO, data) -> int:
    value = memoryview(data).cast("B")
    size = len(value)
    while value:
        written = output.write(value[: 1 << 20])
        if not written:
            raise OSError("bundle write made no progress")
        value = value[written:]
    return size


class _AppendSection(io.RawIOBase):
    """One forward-only view; closing it never closes the shared spool."""

    def __init__(self, output: BinaryIO, start: int, data_start: int) -> None:
        self._output = output
        self._start = start
        self._data_start = data_start

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        if self.closed:
            raise ValueError("section is closed")
        return self._output.tell() - self._start

    def write(self, data) -> int:
        if self.closed:
            raise ValueError("section is closed")
        value = memoryview(data).cast("B")
        size = len(value)
        offset = self._output.tell() - self._data_start
        if size > _MAX_UINT64 - offset:
            raise OverflowError("bundle section table exceeds uint64 range")
        return _write_all(self._output, value)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("section is closed")
        if not self._output.closed:
            self._output.flush()


class BundleWriter:
    """Stage named sections and atomically publish one format-v1 bundle.

    The default uses separate section files. ``enable_direct_staging`` opts
    into one append-only spool with a bounded, whitespace-padded JSON header.
    Both paths preserve section payloads and the same atomic publication API.
    """

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
        self._build_request: Any = None
        self._header_reserve_bytes = 0
        self._direct_path: Path | None = None
        self._direct_output: BinaryIO | None = None
        self._direct_sections: dict[str, dict[str, int]] = {}

    def enable_direct_staging(self, *, header_reserve_bytes: int = 64 * 1024) -> None:
        """Write each section once, without a second full-bundle disk copy.

        Opt in before opening any section. This mode permits only one open
        section at a time, with forward writes and section-relative ``tell``;
        seeking, reading and exposing the spool's descriptor are unsupported.
        The reserved header remains within format v1's existing 100-MiB cap.
        """

        self._ensure_writable()
        if self._section_names or self._header_reserve_bytes:
            raise RuntimeError("direct staging must be selected once before any section")
        if (
            type(header_reserve_bytes) is not int
            or not 0 < header_reserve_bytes <= _MAX_HEADER_SIZE
        ):
            raise ValueError("header reserve must be an integer within the 100 MiB limit")
        self._header_reserve_bytes = header_reserve_bytes

    def _ensure_direct_output(self) -> BinaryIO:
        if self._direct_output is None:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self._destination.name}.",
                suffix=".tmp",
                dir=self._destination.parent,
            )
            self._direct_path = Path(name)
            self._direct_output = os.fdopen(descriptor, "w+b", buffering=0)
            try:
                _write_all(self._direct_output, BUNDLE_MAGIC)
                _write_all(self._direct_output, struct.pack("<Q", self._header_reserve_bytes))
                remaining = self._header_reserve_bytes
                while remaining:
                    size = min(remaining, 1 << 20)
                    _write_all(self._direct_output, b" " * size)
                    remaining -= size
            except BaseException:
                self._failed_section = True
                self._cleanup_staging()
                raise
        return self._direct_output

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
        if self._direct_output is not None:
            self._direct_output.close()
            self._direct_output = None
        if self._direct_path is not None:
            self._direct_path.unlink(missing_ok=True)
            self._direct_path = None
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

    def record_build(self, request: Any) -> None:
        """Describe this bundle's build request in the core provenance section.

        ``finish`` writes the description; bundles without a recorded request
        carry no provenance section.
        """

        self._ensure_writable()
        self._build_request = request

    @contextmanager
    def open_section(self, name: str) -> Iterator[BinaryIO]:
        """Open one file-backed section for incremental binary writes."""

        self._ensure_writable()
        name = _validate_nonempty_string("section name", name)
        if name.startswith(provenance.CORE_SECTION_PREFIX):
            raise ValueError(
                f"section names starting with {provenance.CORE_SECTION_PREFIX!r} "
                "are reserved for the core"
            )
        with self._stage_section(name) as section:
            yield section

    @contextmanager
    def _stage_section(self, name: str) -> Iterator[BinaryIO]:
        self._ensure_writable()
        if name in self._section_names:
            raise ValueError(f"duplicate bundle section name: {name!r}")
        if self._header_reserve_bytes:
            with self._append_section(name) as section:
                yield section
            return

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

    @contextmanager
    def _append_section(self, name: str) -> Iterator[BinaryIO]:
        if self._open_sections:
            raise RuntimeError("direct staging permits only one open section")
        output = self._ensure_direct_output()
        start = output.tell()
        data_start = len(BUNDLE_MAGIC) + 8 + self._header_reserve_bytes
        self._section_names.add(name)
        self._open_sections += 1
        try:
            with _AppendSection(output, start, data_start) as section:
                yield section
            self._direct_sections[name] = {
                "offset": start - data_start,
                "length": output.tell() - start,
            }
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
        if self._build_request is not None:
            record = provenance.collect(self._build_request)
            with self._stage_section(provenance.PROVENANCE_SECTION) as section:
                section.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
            self._build_request = None

        section_table: dict[str, dict[str, int]] = dict(self._direct_sections)
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
        if self._header_reserve_bytes:
            if len(header_bytes) > self._header_reserve_bytes:
                raise ValueError("bundle header exceeds its reserved header capacity")
            output = self._ensure_direct_output()
            try:
                output.seek(len(BUNDLE_MAGIC) + 8)
                _write_all(output, header_bytes)
                output.flush()
                output.close()
                os.replace(self._direct_path, self._destination)
            except BaseException:
                self._failed_section = True
                self._cleanup_staging()
                raise
            self._direct_path = None
            self._direct_output = None
            self._finished = True
            return

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
