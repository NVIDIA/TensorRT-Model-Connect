# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
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
