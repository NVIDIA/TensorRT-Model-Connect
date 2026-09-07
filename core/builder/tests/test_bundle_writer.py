# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect.bundle_writer import (
    BUNDLE_MAGIC,
    BUNDLE_PROVENANCE_MAGIC,
    BundleWriter,
    read_bundle_provenance,
)


def _read_bundle(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    header_size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    header_start = len(BUNDLE_MAGIC) + 8
    header_end = header_start + header_size
    return json.loads(data[header_start:header_end]), data[header_end:]


def _write_raw_bundle(
    path: Path, header: object, *, payload: bytes = b"", provenance: object | None = None
) -> None:
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_provenance = json.dumps(
        provenance if provenance is not None else {"format": 1},
        separators=(",", ":"),
    ).encode()
    path.write_bytes(
        BUNDLE_MAGIC
        + struct.pack("<Q", len(raw_header))
        + raw_header
        + payload
        + raw_provenance
        + struct.pack("<Q", len(raw_provenance))
        + BUNDLE_PROVENANCE_MAGIC
    )


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


def test_writer_appends_provenance_outside_family_sections(tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    provenance = {
        "format": 1,
        "checkpoint": {"id": "example/model", "revision": "b" * 40},
        "build": {"source_revision": "a" * 40},
        "request": {},
    }
    writer = BundleWriter(destination)
    writer.set_header(family="family", task="text_generation", backend="trt")
    writer.add_bytes("engine.plan", b"plan")
    writer.set_provenance(provenance)

    writer.finish()

    header, payload_and_trailer = _read_bundle(destination)
    raw_provenance = json.dumps(provenance, separators=(",", ":")).encode()
    assert header["sections"] == {"engine.plan": {"offset": 0, "length": 4}}
    assert "provenance.json" not in header["sections"]
    assert payload_and_trailer == (
        b"plan"
        + raw_provenance
        + struct.pack("<Q", len(raw_provenance))
        + BUNDLE_PROVENANCE_MAGIC
    )
    assert read_bundle_provenance(destination) == provenance


@pytest.mark.parametrize(
    "header",
    [
        {
            "format": 1,
            "family": "family",
            "task": "text_generation",
            "backend": "trt",
        },
        {
            "format": 1,
            "family": "family",
            "task": "text_generation",
            "backend": "trt",
            "sections": {},
            "model_id": "legacy",
        },
        {
            "format": 1,
            "family": "family",
            "task": "text_generation",
            "backend": "trt",
            "sections": {"engine.plan": {"offset": 0}},
        },
        {
            "format": 1,
            "family": "family",
            "task": "text_generation",
            "backend": "trt",
            "sections": {"engine.plan": {"offset": 0, "length": 1}},
        },
    ],
    ids=("missing-sections", "unsupported-field", "incomplete-section", "section-bounds"),
)
def test_provenance_reader_rejects_headers_the_runtime_rejects(
    tmp_path: Path, header: object
) -> None:
    destination = tmp_path / "model.bundle"
    _write_raw_bundle(destination, header)

    with pytest.raises(ValueError):
        read_bundle_provenance(destination)


def test_provenance_reader_rejects_a_non_object_trailer(tmp_path: Path) -> None:
    destination = tmp_path / "model.bundle"
    _write_raw_bundle(
        destination,
        {
            "format": 1,
            "family": "family",
            "task": "text_generation",
            "backend": "trt",
            "sections": {},
        },
        provenance=[],
    )

    with pytest.raises(ValueError, match="JSON object"):
        read_bundle_provenance(destination)


def test_provenance_must_be_one_json_object(tmp_path: Path) -> None:
    writer = BundleWriter(tmp_path / "model.bundle")
    with pytest.raises(TypeError, match="JSON object"):
        writer.set_provenance([])
    writer.set_provenance({"format": 1})
    with pytest.raises(RuntimeError, match="already set"):
        writer.set_provenance({"format": 1})
    writer.abort()


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
def test_writer_rejects_unsafe_header_ids(
    tmp_path: Path, field: str, value: str
) -> None:
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


def test_failed_atomic_replace_preserves_destination(
    monkeypatch, tmp_path: Path
) -> None:
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
