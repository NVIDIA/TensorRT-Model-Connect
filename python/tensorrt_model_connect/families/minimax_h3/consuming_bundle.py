# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-resumable, source-consuming bundle finalization for MiniMax-H3.

This is a build-time helper.  It deliberately leaves the ordinary bundle
writer unchanged: H3 is unusually large, so its already-qualified plan files
are copied into a stable same-directory partial bundle one at a time and are
removed only after the copied range and an atomic journal commit are durable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, BundleInfo


_SCHEMA_VERSION = 1
_MAX_BUNDLE_HEADER_SIZE = 100 * 1024 * 1024
_MAX_JOURNAL_SIZE = 1 << 20
_COPY_CHUNK_BYTES = 8 << 20
_SHA256_CHARS = frozenset("0123456789abcdef")


class ConsumingBundleError(RuntimeError):
    """The resumable H3 bundle assembly cannot be advanced safely."""


@dataclass(frozen=True)
class ConsumingBundleSection:
    """One exact section and, optionally, the source to consume after commit."""

    name: str
    size: int
    sha256: str
    source_path: Path | None = None
    data: bytes | None = None
    consume_source: bool = False

    @classmethod
    def from_file(
        cls,
        name: str,
        source_path: str | Path,
        *,
        size: int,
        sha256: str,
        consume_source: bool,
    ) -> ConsumingBundleSection:
        return cls(
            name=name,
            size=size,
            sha256=sha256,
            source_path=Path(source_path),
            consume_source=consume_source,
        )

    @classmethod
    def from_bytes(cls, name: str, data: bytes) -> ConsumingBundleSection:
        payload = bytes(data)
        return cls(
            name=name,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            data=payload,
        )


@dataclass(frozen=True)
class _SectionLayout:
    section: ConsumingBundleSection
    offset: int


@dataclass(frozen=True)
class _BundleLayout:
    header: bytes
    data_start: int
    sections: tuple[_SectionLayout, ...]
    payload_size: int
    assembly_sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
    )


def assembly_paths(destination: str | Path) -> tuple[Path, Path]:
    """Return stable same-directory partial and journal paths."""

    path = Path(destination)
    partial = path.with_name(f".{path.name}.partial")
    journal = path.with_name(f".{path.name}.partial.json")
    return partial, journal


def _entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists without following symlinks."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ConsumingBundleError(f"Unable to inspect H3 bundle path: {path}") from error
    return True


def _paths_alias(left: Path, right: Path) -> bool:
    left_name = os.path.normcase(os.path.abspath(os.fspath(left)))
    right_name = os.path.normcase(os.path.abspath(os.fspath(right)))
    if left_name == right_name:
        return True
    try:
        return _entry_exists(left) and _entry_exists(right) and os.path.samefile(left, right)
    except OSError:
        return False


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _validate_sections(
    sections: Sequence[ConsumingBundleSection],
) -> tuple[_SectionLayout, ...]:
    names: set[str] = set()
    result: list[_SectionLayout] = []
    offset = 0
    for section in sections:
        if not section.name or section.name in names:
            raise ValueError(f"Invalid or duplicate bundle section name: {section.name!r}")
        names.add(section.name)
        if (
            not isinstance(section.size, int)
            or isinstance(section.size, bool)
            or section.size <= 0
        ):
            raise ValueError(f"Bundle section {section.name!r} must have a positive size")
        if not _is_sha256(section.sha256):
            raise ValueError(f"Bundle section {section.name!r} has an invalid SHA-256")
        sources = int(section.source_path is not None) + int(section.data is not None)
        if sources != 1:
            raise ValueError(
                f"Bundle section {section.name!r} must have exactly one data source"
            )
        if section.consume_source and section.source_path is None:
            raise ValueError(
                f"In-memory bundle section {section.name!r} cannot consume a source"
            )
        if section.data is not None:
            if len(section.data) != section.size:
                raise ValueError(f"Bundle section {section.name!r} has the wrong byte size")
            if hashlib.sha256(section.data).hexdigest() != section.sha256:
                raise ValueError(f"Bundle section {section.name!r} has the wrong SHA-256")
        result.append(_SectionLayout(section=section, offset=offset))
        offset += section.size
    return tuple(result)


def _header_bytes(info: BundleInfo, sections: tuple[_SectionLayout, ...]) -> bytes:
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
        "sections": {
            item.section.name: {
                "offset": item.offset,
                "size": item.section.size,
            }
            for item in sections
        },
    }
    encoded = json.dumps(header, indent=2).encode("utf-8")
    if len(encoded) > _MAX_BUNDLE_HEADER_SIZE:
        raise ValueError(
            f"Bundle JSON header exceeds the 100 MiB runtime limit: {len(encoded)} bytes"
        )
    return encoded


def _layout(info: BundleInfo, sections: Sequence[ConsumingBundleSection]) -> _BundleLayout:
    section_layouts = _validate_sections(sections)
    header = _header_bytes(info, section_layouts)
    identity = {
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "sections": [
            {
                "name": item.section.name,
                "offset": item.offset,
                "size": item.section.size,
                "sha256": item.section.sha256,
                "consume_source": item.section.consume_source,
            }
            for item in section_layouts
        ],
    }
    assembly_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _BundleLayout(
        header=header,
        data_start=16 + len(header),
        sections=section_layouts,
        payload_size=sum(item.section.size for item in section_layouts),
        assembly_sha256=assembly_sha256,
    )


def _journal_core(layout: _BundleLayout, committed_count: int) -> dict[str, int | str]:
    committed_bytes = sum(
        item.section.size for item in layout.sections[:committed_count]
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "assembly_sha256": layout.assembly_sha256,
        "committed_count": committed_count,
        "committed_payload_bytes": committed_bytes,
    }


def _journal_value(layout: _BundleLayout, committed_count: int) -> dict[str, int | str]:
    result = _journal_core(layout, committed_count)
    result["state_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _sync_file(path: Path) -> None:
    # Windows' CRT rejects ``_commit`` on a read-only descriptor, so request a
    # writable handle even though this operation changes no bytes.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _sync_parent_directory(path: Path) -> None:
    """Persist a same-directory rename/unlink where directory fsync is available."""

    if os.name == "nt":
        # ``os.fsync`` above maps to FlushFileBuffers.  CPython cannot open an
        # NTFS directory as a normal file descriptor, while same-volume
        # ``os.replace`` still supplies the required atomic namespace switch.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_file(path)
        _sync_parent_directory(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_journal(path: Path, layout: _BundleLayout) -> int | None:
    if not _entry_exists(path):
        return None
    try:
        journal_stat = path.lstat()
    except OSError as error:
        raise ConsumingBundleError(f"Unable to inspect H3 bundle journal: {path}") from error
    if not stat.S_ISREG(journal_stat.st_mode):
        raise ConsumingBundleError(f"H3 bundle journal is not a regular file: {path}")
    if journal_stat.st_size <= 0 or journal_stat.st_size > _MAX_JOURNAL_SIZE:
        return None

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate journal key")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    expected_keys = {
        "schema_version",
        "assembly_sha256",
        "committed_count",
        "committed_payload_bytes",
        "state_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    schema_version = value.get("schema_version")
    assembly_sha256 = value.get("assembly_sha256")
    raw_committed_count = value.get("committed_count")
    committed_payload_bytes = value.get("committed_payload_bytes")
    state_sha256 = value.get("state_sha256")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _SCHEMA_VERSION
        or not _is_sha256(assembly_sha256)
        or not isinstance(raw_committed_count, int)
        or isinstance(raw_committed_count, bool)
        or raw_committed_count < 0
        or raw_committed_count > len(layout.sections)
        or not isinstance(committed_payload_bytes, int)
        or isinstance(committed_payload_bytes, bool)
        or committed_payload_bytes < 0
        or committed_payload_bytes > layout.payload_size
        or not _is_sha256(state_sha256)
    ):
        return None
    unsigned_value = {key: item for key, item in value.items() if key != "state_sha256"}
    actual_state_sha256 = hashlib.sha256(
        json.dumps(unsigned_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if state_sha256 != actual_state_sha256:
        return None
    committed_count = raw_committed_count
    if unsigned_value != _journal_core(layout, committed_count):
        return None
    return committed_count


def _hash_range(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise ConsumingBundleError(f"Bundle range is truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _truncate_and_sync(path: Path, size: int) -> None:
    with path.open("r+b") as stream:
        stream.truncate(size)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_partial_header(path: Path, layout: _BundleLayout) -> int:
    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise ConsumingBundleError(f"H3 bundle file is not regular: {path}")
        size = path_stat.st_size
        with path.open("rb") as stream:
            if stream.read(8) != BUNDLE_MAGIC:
                raise ConsumingBundleError(f"H3 partial bundle has invalid magic: {path}")
            raw_header_size = stream.read(8)
            if len(raw_header_size) != 8:
                raise ConsumingBundleError(f"H3 partial bundle has a truncated header: {path}")
            header_size = struct.unpack("<Q", raw_header_size)[0]
            if header_size != len(layout.header) or stream.read(header_size) != layout.header:
                raise ConsumingBundleError(
                    f"H3 partial bundle belongs to a different assembly: {path}"
                )
    except OSError as error:
        raise ConsumingBundleError(f"Unable to validate H3 partial bundle: {path}") from error
    if size < layout.data_start:
        raise ConsumingBundleError(f"H3 partial bundle ends inside its header: {path}")
    return size


def _record_end(layout: _BundleLayout, count: int) -> int:
    return layout.data_start + sum(
        item.section.size for item in layout.sections[:count]
    )


def _source_is_exact(section: ConsumingBundleSection) -> bool:
    if section.data is not None:
        return True
    try:
        return _validated_source_identity(section) is not None
    except (ConsumingBundleError, OSError):
        return False


def _open_exact_source(
    section: ConsumingBundleSection,
) -> tuple[object, _FileIdentity]:
    source_path = section.source_path
    if source_path is None:
        raise AssertionError("file-backed section has no source path")
    try:
        before_stat = source_path.lstat()
    except OSError as error:
        raise ConsumingBundleError(
            f"H3 bundle section source is unavailable: {source_path}"
        ) from error
    before = _file_identity(before_stat)
    if not stat.S_ISREG(before_stat.st_mode) or before.size != section.size:
        raise ConsumingBundleError(
            f"H3 bundle section source has the wrong type or size: {source_path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(source_path, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or _file_identity(opened_stat) != before:
            raise ConsumingBundleError(
                f"H3 bundle section source changed before reading: {source_path}"
            )
        source = os.fdopen(descriptor, "rb")
        descriptor = None
        return source, before
    except OSError as error:
        raise ConsumingBundleError(
            f"Unable to open H3 bundle section source safely: {source_path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validated_source_identity(
    section: ConsumingBundleSection,
) -> _FileIdentity | None:
    if section.source_path is None:
        return None
    source, identity = _open_exact_source(section)
    digest = hashlib.sha256()
    remaining = section.size
    with source:
        while remaining:
            chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise ConsumingBundleError(
                    f"H3 bundle section source changed while hashing: {section.source_path}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise ConsumingBundleError(
                f"H3 bundle section source grew while hashing: {section.source_path}"
            )
        if _file_identity(os.fstat(source.fileno())) != identity:
            raise ConsumingBundleError(
                f"H3 bundle section source changed while hashing: {section.source_path}"
            )
    if digest.hexdigest() != section.sha256:
        raise ConsumingBundleError(
            f"H3 bundle section source failed SHA-256 validation: {section.source_path}"
        )
    try:
        after = section.source_path.lstat()
    except OSError as error:
        raise ConsumingBundleError(
            f"H3 bundle section source disappeared after hashing: {section.source_path}"
        ) from error
    if not stat.S_ISREG(after.st_mode) or _file_identity(after) != identity:
        raise ConsumingBundleError(
            f"H3 bundle section source changed after hashing: {section.source_path}"
        )
    return identity


def _reconstruct_journal(path: Path, layout: _BundleLayout, file_size: int) -> int:
    """Recover the largest exact prefix when the small journal is unavailable."""

    count = 0
    for item in layout.sections:
        start = layout.data_start + item.offset
        end = start + item.section.size
        if file_size < end:
            break
        if _hash_range(path, start, item.section.size) != item.section.sha256:
            if not _source_is_exact(item.section):
                raise ConsumingBundleError(
                    "H3 partial bundle has a corrupt range whose source is unavailable: "
                    f"{item.section.name}"
                )
            break
        count += 1
    expected_end = _record_end(layout, count)
    if file_size != expected_end:
        if count < len(layout.sections) and not _source_is_exact(
            layout.sections[count].section
        ):
            raise ConsumingBundleError(
                "H3 partial bundle has an uncommitted tail whose source is unavailable: "
                f"{layout.sections[count].section.name}"
            )
        _truncate_and_sync(path, expected_end)
    return count


def _resume_prefix(path: Path, layout: _BundleLayout, committed_count: int) -> int:
    file_size = _validate_partial_header(path, layout)
    for item in layout.sections[:committed_count]:
        actual = _hash_range(
            path,
            layout.data_start + item.offset,
            item.section.size,
        )
        if actual != item.section.sha256:
            raise ConsumingBundleError(
                f"H3 committed bundle range failed SHA-256 validation: {item.section.name}"
            )
    committed_end = _record_end(layout, committed_count)
    if file_size < committed_end:
        raise ConsumingBundleError("H3 partial bundle is shorter than its committed journal")

    recovered_count = committed_count
    for item in layout.sections[committed_count:]:
        end = layout.data_start + item.offset + item.section.size
        if file_size < end:
            break
        actual = _hash_range(
            path,
            layout.data_start + item.offset,
            item.section.size,
        )
        if actual != item.section.sha256:
            break
        recovered_count += 1

    recovered_end = _record_end(layout, recovered_count)
    if file_size > recovered_end:
        if recovered_count < len(layout.sections) and not _source_is_exact(
            layout.sections[recovered_count].section
        ):
            raise ConsumingBundleError(
                "H3 partial bundle has an uncommitted tail whose source is unavailable: "
                f"{layout.sections[recovered_count].section.name}"
            )
        _truncate_and_sync(path, recovered_end)
    return recovered_count


def _append_file(
    path: Path,
    start: int,
    section: ConsumingBundleSection,
) -> _FileIdentity:
    source_path = section.source_path
    if source_path is None:
        raise AssertionError("file-backed section has no source path")
    source, identity = _open_exact_source(section)

    digest = hashlib.sha256()
    try:
        with source, path.open("r+b") as output:
            output.seek(0, os.SEEK_END)
            if output.tell() != start:
                raise ConsumingBundleError("H3 partial bundle append offset changed unexpectedly")
            remaining = section.size
            while remaining:
                chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ConsumingBundleError(
                        f"H3 bundle section source changed while reading: {source_path}"
                    )
                written = output.write(chunk)
                if written != len(chunk):
                    raise OSError(f"short bundle write: {written}/{len(chunk)}")
                digest.update(chunk)
                remaining -= len(chunk)
            if source.read(1):
                raise ConsumingBundleError(
                    f"H3 bundle section source grew while reading: {source_path}"
                )
            if digest.hexdigest() != section.sha256:
                raise ConsumingBundleError(
                    f"H3 bundle section source failed SHA-256 validation: {source_path}"
                )
            if _file_identity(os.fstat(source.fileno())) != identity:
                raise ConsumingBundleError(
                    f"H3 bundle section source changed while reading: {source_path}"
                )
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        _truncate_and_sync(path, start)
        raise
    return identity


def _append_bytes(path: Path, start: int, section: ConsumingBundleSection) -> None:
    if section.data is None:
        raise AssertionError("in-memory section has no data")
    try:
        with path.open("r+b") as output:
            output.seek(0, os.SEEK_END)
            if output.tell() != start:
                raise ConsumingBundleError("H3 partial bundle append offset changed unexpectedly")
            written = output.write(section.data)
            if written != section.size:
                raise OSError(f"short bundle write: {written}/{section.size}")
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        _truncate_and_sync(path, start)
        raise


def _unlink_exact_source(
    section: ConsumingBundleSection,
    *,
    expected_identity: _FileIdentity | None = None,
    mismatch_is_error: bool = True,
) -> None:
    if not section.consume_source:
        return
    path = section.source_path
    if path is None or not _entry_exists(path):
        return
    try:
        identity = (
            _validated_source_identity(section)
            if expected_identity is None
            else expected_identity
        )
        before_unlink = path.lstat()
    except (ConsumingBundleError, OSError) as error:
        if not mismatch_is_error:
            return
        raise ConsumingBundleError(f"Refusing to remove changed H3 bundle source: {path}") from error
    if (
        identity is None
        or not stat.S_ISREG(before_unlink.st_mode)
        or _file_identity(before_unlink) != identity
    ):
        if not mismatch_is_error:
            return
        raise ConsumingBundleError(f"Refusing to remove changed H3 bundle source: {path}")
    path.unlink()
    _sync_parent_directory(path)


def _validate_complete(path: Path, layout: _BundleLayout) -> bool:
    if not _entry_exists(path):
        return False
    try:
        file_size = _validate_partial_header(path, layout)
        if file_size != layout.data_start + layout.payload_size:
            return False
        for item in layout.sections:
            if (
                _hash_range(
                    path,
                    layout.data_start + item.offset,
                    item.section.size,
                )
                != item.section.sha256
            ):
                return False
        return True
    except (ConsumingBundleError, OSError):
        return False


def _invoke_failure(
    failure_injector: Callable[[str], None] | None,
    event: str,
) -> None:
    if failure_injector is not None:
        failure_injector(event)


def _remove_regular_journal(path: Path) -> None:
    if not _entry_exists(path):
        return
    try:
        journal_stat = path.lstat()
    except OSError as error:
        raise ConsumingBundleError(f"Unable to inspect H3 bundle journal: {path}") from error
    if not stat.S_ISREG(journal_stat.st_mode):
        raise ConsumingBundleError(f"H3 bundle journal is not a regular file: {path}")
    path.unlink()
    _sync_parent_directory(path)


def write_consuming_bundle(
    destination: str | Path,
    info: BundleInfo,
    sections: Sequence[ConsumingBundleSection],
    *,
    failure_injector: Callable[[str], None] | None = None,
) -> Path:
    """Finalize one exact H3 bundle while consuming committed file sources.

    Existing destination files remain untouched until the same-directory
    partial has passed a complete header, size, and per-section SHA-256 audit.
    """

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    layout = _layout(info, sections)
    partial, journal = assembly_paths(output)
    file_sources = [
        item.section.source_path
        for item in layout.sections
        if item.section.source_path is not None
    ]
    for index, source in enumerate(file_sources):
        if any(_paths_alias(source, reserved) for reserved in (output, partial, journal)):
            raise ConsumingBundleError(
                f"H3 bundle source aliases an assembly artifact: {source}"
            )
        if any(_paths_alias(source, previous) for previous in file_sources[:index]):
            raise ConsumingBundleError(
                f"H3 bundle sections reuse the same source file: {source}"
            )

    if _entry_exists(output):
        try:
            output_stat = output.lstat()
        except OSError as error:
            raise ConsumingBundleError(f"Unable to inspect H3 bundle destination: {output}") from error
        if not stat.S_ISREG(output_stat.st_mode):
            raise ConsumingBundleError(
                f"H3 bundle destination is not a regular file: {output}"
            )

    if _validate_complete(output, layout):
        if _entry_exists(journal):
            journal_stat = journal.lstat()
            if not stat.S_ISREG(journal_stat.st_mode):
                raise ConsumingBundleError(
                    f"H3 bundle journal is not a regular file: {journal}"
                )
        if _entry_exists(partial):
            if not _validate_complete(partial, layout):
                raise ConsumingBundleError(
                    "Completed H3 bundle has an unrelated same-name partial assembly"
                )
            partial.unlink()
            _sync_parent_directory(partial)
        _remove_regular_journal(journal)
        for item in layout.sections:
            _unlink_exact_source(item.section, mismatch_is_error=False)
        return output

    if _entry_exists(partial):
        file_size = _validate_partial_header(partial, layout)
        committed_count = _read_journal(journal, layout)
        if committed_count is None:
            committed_count = _reconstruct_journal(partial, layout, file_size)
            _atomic_write_json(journal, _journal_value(layout, committed_count))
        else:
            recovered_count = _resume_prefix(partial, layout, committed_count)
            if recovered_count != committed_count:
                committed_count = recovered_count
                _atomic_write_json(journal, _journal_value(layout, committed_count))
    else:
        if _entry_exists(journal):
            raise ConsumingBundleError(
                "H3 bundle journal exists without its partial bundle"
            )
        with partial.open("xb") as stream:
            stream.write(BUNDLE_MAGIC)
            stream.write(struct.pack("<Q", len(layout.header)))
            stream.write(layout.header)
            stream.flush()
            os.fsync(stream.fileno())
        _invoke_failure(failure_injector, "after_partial_header_fsync")
        committed_count = 0
        _atomic_write_json(journal, _journal_value(layout, committed_count))

    for item in layout.sections[:committed_count]:
        _unlink_exact_source(item.section)

    for index in range(committed_count, len(layout.sections)):
        item = layout.sections[index]
        start = layout.data_start + item.offset
        source_identity = None
        if item.section.source_path is not None:
            source_identity = _append_file(partial, start, item.section)
        else:
            _append_bytes(partial, start, item.section)
        _invoke_failure(
            failure_injector,
            f"after_section_write_fsync:{item.section.name}",
        )
        actual = _hash_range(partial, start, item.section.size)
        if actual != item.section.sha256:
            _truncate_and_sync(partial, start)
            raise ConsumingBundleError(
                f"H3 written bundle range failed SHA-256 validation: {item.section.name}"
            )
        _invoke_failure(
            failure_injector,
            f"after_section_range_verify:{item.section.name}",
        )
        committed_count = index + 1
        _atomic_write_json(journal, _journal_value(layout, committed_count))
        _invoke_failure(
            failure_injector,
            f"after_journal_commit:{item.section.name}",
        )
        _unlink_exact_source(item.section, expected_identity=source_identity)
        _invoke_failure(
            failure_injector,
            f"after_source_unlink:{item.section.name}",
        )

    if not _validate_complete(partial, layout):
        raise ConsumingBundleError("Completed H3 partial bundle failed its final audit")
    _invoke_failure(failure_injector, "before_final_replace")
    os.replace(partial, output)
    _sync_file(output)
    _sync_parent_directory(output)
    _invoke_failure(failure_injector, "after_final_replace")
    _remove_regular_journal(journal)
    return output
