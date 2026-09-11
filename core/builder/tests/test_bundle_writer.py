# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, BundleWriter


def _read_bundle(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    header_size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    header_start = len(BUNDLE_MAGIC) + 8
    header_end = header_start + header_size
    return json.loads(data[header_start:header_end]), data[header_end:]


def test_writer_streams_sections_and_emits_only_the_fixed_header(tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    writer = BundleWriter(destination)
    writer.set_header(family="gpt_neo", task="text_generation", backend="trt")
    with writer.open_section("engine.plan") as section:
        section.write(b"engine-")
        section.write(b"bytes")
    writer.add_json("config.json", {"size": 7})
    writer.add_bytes("tokenizer.model", b"tokens")

    writer.finish()

    header, payload = _read_bundle(destination)
    assert list(header) == ["format", "family", "task", "backend", "sections"]
    assert header == {
        "format": 1,
        "family": "gpt_neo",
        "task": "text_generation",
        "backend": "trt",
        "sections": {
            "engine.plan": {"offset": 0, "length": 12},
            "config.json": {"offset": 12, "length": 10},
            "tokenizer.model": {"offset": 22, "length": 6},
        },
    }
    assert payload == b'engine-bytes{"size":7}tokens'


def test_writer_rejects_duplicate_and_empty_section_names(tmp_path: Path) -> None:
    writer = BundleWriter(tmp_path / "model.bundle")
    writer.add_bytes("engine.plan", b"one")

    with pytest.raises(ValueError, match="duplicate"):
        writer.add_bytes("engine.plan", b"two")
    with pytest.raises(ValueError, match="section name"):
        writer.add_bytes("", b"two")

    writer.abort()


@pytest.mark.parametrize(
    ("field", "value"),
    [("family", "../family"), ("task", ""), ("backend", "")],
)
def test_writer_rejects_unsafe_header_ids(tmp_path: Path, field: str, value: str) -> None:
    header = {"family": "family", "task": "text_generation", "backend": "trt"}
    header[field] = value
    writer = BundleWriter(tmp_path / "model.bundle")

    with pytest.raises(ValueError, match=field):
        writer.set_header(**header)


def test_writer_requires_an_existing_output_directory(tmp_path: Path) -> None:
    destination = tmp_path / "missing" / "model.bundle"

    with pytest.raises(FileNotFoundError, match="output directory does not exist"):
        BundleWriter(destination)


def test_header_must_be_set_once_before_finish(tmp_path: Path) -> None:
    writer = BundleWriter(tmp_path / "model.bundle")
    with pytest.raises(RuntimeError, match="not set"):
        writer.finish()

    writer.set_header(family="family", task="text_generation", backend="trt")
    with pytest.raises(RuntimeError, match="already set"):
        writer.set_header(family="family", task="text_generation", backend="trt")

    writer.abort()


def test_abort_discards_staging_and_preserves_destination(tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_bytes("engine.plan", b"new")

    writer.abort()

    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]
    with pytest.raises(RuntimeError, match="aborted"):
        writer.finish()


def test_failed_atomic_replace_preserves_destination(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_bytes("engine.plan", b"new")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        writer.finish()
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".model.bundle.*.tmp"))

    writer.abort()


def test_borrowed_file_mixes_with_streamed_sections_without_staging_copy(tmp_path: Path) -> None:
    source = tmp_path / "existing.plan"
    source.write_bytes(b"engine-bytes")
    before = source.stat()
    destination = tmp_path / "model.bundle"
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_file("engine.plan", source)
    assert list(tmp_path.iterdir()) == [source]
    writer.add_json("config.json", {"size": 7})
    with writer.open_section("tokenizer.model") as section:
        section.write(b"tokens")
    writer.finish()

    header, payload = _read_bundle(destination)
    assert header["sections"] == {
        "engine.plan": {"offset": 0, "length": 12},
        "config.json": {"offset": 12, "length": 10},
        "tokenizer.model": {"offset": 22, "length": 6},
    }
    assert payload == b'engine-bytes{"size":7}tokens'
    assert source.read_bytes() == b"engine-bytes"
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert set(tmp_path.iterdir()) == {source, destination}
    with pytest.raises(RuntimeError, match="already finished"):
        writer.add_file("other", source)


def test_borrowed_files_reuse_section_guards_and_require_external_files(tmp_path: Path) -> None:
    source = tmp_path / "existing.plan"
    source.write_bytes(b"plan")
    writer = BundleWriter(tmp_path / "model.bundle")
    writer.add_file("borrowed", source)
    with pytest.raises(ValueError, match="duplicate"):
        writer.add_bytes("borrowed", b"other")
    writer.add_bytes("staged", b"bytes")
    with pytest.raises(ValueError, match="duplicate"):
        writer.add_file("staged", source)
    with pytest.raises(ValueError, match="section name"):
        writer.add_file("", source)
    for missing in (tmp_path, tmp_path / "missing.plan"):
        with pytest.raises(FileNotFoundError, match="source is not a file"):
            writer.add_file("invalid", missing)
    with writer.open_section("owned") as section:
        section.write(b"owned")
        owned_path = Path(section.name)
    with pytest.raises(ValueError, match="writer-owned paths"):
        writer.add_file("invalid", owned_path)
    writer.abort()
    assert source.read_bytes() == b"plan"
    with pytest.raises(RuntimeError, match="aborted"):
        writer.add_file("other", source)


def test_borrowing_destination_is_rejected_without_modifying_it(tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    writer = BundleWriter(destination)
    with pytest.raises(ValueError, match="writer-owned paths"):
        writer.add_file("previous", destination)
    writer.abort()
    assert destination.read_bytes() == b"previous"


@pytest.mark.parametrize("failure", ("abort", "copy", "replace"))
def test_borrowed_source_and_old_destination_survive_abort_or_finish_failure(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    source, destination = tmp_path / "existing.plan", tmp_path / "model.bundle"
    source.write_bytes(b"plan")
    destination.write_bytes(b"previous")
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_file("engine.plan", source)
    writer.add_bytes("metadata", b"metadata")

    def fail(*_args, **_kwargs):
        raise OSError("publication failed")

    if failure != "abort":
        monkeypatch.setattr(
            shutil if failure == "copy" else os,
            "copyfileobj" if failure == "copy" else "replace",
            fail,
        )
        with pytest.raises(OSError, match="publication failed"):
            writer.finish()
        assert not list(tmp_path.glob(".model.bundle.*.tmp"))
    writer.abort()
    assert source.read_bytes() == b"plan"
    assert destination.read_bytes() == b"previous"
    assert set(tmp_path.iterdir()) == {source, destination}


@pytest.mark.parametrize("change", ("size", "mtime", "replacement"))
def test_changed_borrowed_source_fails_before_publishing(tmp_path: Path, change: str) -> None:
    source, destination = tmp_path / "existing.plan", tmp_path / "model.bundle"
    source.write_bytes(b"original")
    destination.write_bytes(b"previous")
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_file("engine.plan", source)
    before = source.stat()
    if change == "size":
        source.write_bytes(b"longer source content")
    elif change == "mtime":
        source.write_bytes(b"modified")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        replacement = tmp_path / "replacement.plan"
        replacement.write_bytes(b"modified")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, source)
    with pytest.raises(RuntimeError, match="borrowed bundle section source changed"):
        writer.finish()
    writer.abort()
    assert source.is_file()
    assert destination.read_bytes() == b"previous"
    assert set(tmp_path.iterdir()) == {source, destination}
