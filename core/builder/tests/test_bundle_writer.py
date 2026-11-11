# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect import bundle_writer as owner
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


# Opt-in single-file staging retains the same payload and publication contracts.

def read(path):
    with path.open("rb") as handle:
        assert handle.read(8) == BUNDLE_MAGIC
        length = struct.unpack("<Q", handle.read(8))[0]
        header = handle.read(length)
        return json.loads(header), handle.read(), length, header


def writer(path, *, reserve=65536):
    result = BundleWriter(path)
    result.enable_direct_staging(header_reserve_bytes=reserve)
    result.set_header(family="fixture", task="text_generation", backend="trt")
    return result


def test_direct_and_legacy_keep_identical_header_semantics_and_every_payload_byte(tmp_path):
    old, new = tmp_path / "old.bundle", tmp_path / "new.bundle"
    legacy, direct = BundleWriter(old), writer(new)
    legacy.set_header(family="fixture", task="text_generation", backend="trt")
    for target in (legacy, direct):
        target.add_bytes("engine.plan", bytes(range(256)) * 100)
        target.add_bytes("empty.bin", b"")
        target.add_json("config.json", {"text": "\u8fd0\u884c", "version": 1})
        with target.open_section("last.bin") as section:
            assert section.tell() == 0
            section.write(b"first")
            assert section.tell() == 5
            section.write(b"last")
        target.finish()
    before, after = read(old), read(new)
    assert before[:2] == after[:2]
    assert after[2] == 65536
    assert after[3].endswith(b" ") and len(after[3].rstrip()) < 65536
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.bundle", "old.bundle"]


def test_direct_stage_uses_one_spool_and_renames_the_same_inode_without_copying(
    monkeypatch, tmp_path
):
    destination = tmp_path / "model.bundle"
    direct = writer(destination)
    chunk = b"exact model bytes" * (1 << 16)
    for index in range(8):
        direct.add_bytes(f"part-{index}.plan", chunk)
        files = list(tmp_path.iterdir())
        assert len(files) == 1 and files[0].is_file() and not destination.exists()
        assert files[0].stat().st_size == 16 + 65536 + (index + 1) * len(chunk)
    spool = files[0]
    identity = spool.stat().st_ino

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct publication must not copy section payloads")

    monkeypatch.setattr(owner.shutil, "copyfileobj", forbidden)
    direct.finish()
    assert destination.stat().st_ino == identity and not spool.exists()
    assert read(destination)[1] == chunk * 8


@pytest.mark.parametrize("reserve", [True, False, 0, -1, 1.5, "65536", 100 * 1024**2 + 1])
def test_direct_header_reserve_is_explicit_and_bounded(tmp_path, reserve):
    direct = BundleWriter(tmp_path / "model.bundle")
    with pytest.raises(ValueError, match="reserve"):
        direct.enable_direct_staging(header_reserve_bytes=reserve)
    assert not list(tmp_path.iterdir())


def test_reserve_exhaustion_does_not_replace_destination(tmp_path):
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    direct = writer(destination, reserve=1)
    direct.add_bytes("new.plan", b"new")
    with pytest.raises(ValueError, match="reserved header"):
        direct.finish()
    assert destination.read_bytes() == b"previous"
    direct.abort()
    assert list(tmp_path.iterdir()) == [destination]


def test_direct_is_selected_once_and_before_any_section(tmp_path):
    direct = writer(tmp_path / "direct.bundle")
    with pytest.raises(RuntimeError, match="once"):
        direct.enable_direct_staging()
    legacy = BundleWriter(tmp_path / "legacy.bundle")
    legacy.add_bytes("existing.plan", b"old")
    with pytest.raises(RuntimeError, match="before"):
        legacy.enable_direct_staging()
    direct.abort()
    legacy.abort()


def test_direct_sections_are_forward_only_exclusive_and_closed_at_context_end(tmp_path):
    direct = writer(tmp_path / "model.bundle")
    with direct.open_section("first.plan") as section:
        with pytest.raises(io.UnsupportedOperation):
            section.seek(0)
        with pytest.raises(io.UnsupportedOperation):
            section.fileno()
        with pytest.raises(RuntimeError, match="one open"):
            with direct.open_section("nested.plan"):
                pass
        with pytest.raises(RuntimeError, match="section is open"):
            direct.finish()
        section.write(b"first")
    with pytest.raises(ValueError, match="closed"):
        section.write(b"cannot change a closed section")
    direct.add_bytes("nested.plan", b"second")
    direct.finish()
    assert read(tmp_path / "model.bundle")[1] == b"firstsecond"


def test_failed_direct_section_prevents_publication_and_abort_removes_only_owned_spool(tmp_path):
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    unrelated = tmp_path / "other.plan"
    unrelated.write_bytes(b"keep")
    direct = writer(destination)
    with pytest.raises(OSError, match="producer failed"):
        with direct.open_section("failed.plan") as section:
            section.write(b"partial")
            raise OSError("producer failed")
    with pytest.raises(RuntimeError, match="write failed"):
        direct.finish()
    direct.abort()
    direct.abort()
    assert destination.read_bytes() == b"previous" and unrelated.read_bytes() == b"keep"
    assert len(list(tmp_path.iterdir())) == 2


def test_failed_direct_publication_preserves_destination_and_cleans_its_spool(
    monkeypatch, tmp_path
):
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    direct = writer(destination)
    direct.add_bytes("new.plan", b"new")

    def fail(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        direct.finish()
    assert destination.read_bytes() == b"previous" and list(tmp_path.iterdir()) == [destination]
    with pytest.raises(RuntimeError, match="write failed"):
        direct.finish()
    direct.abort()


def test_direct_payload_overflow_is_rejected_before_writing(monkeypatch, tmp_path):
    direct = writer(tmp_path / "model.bundle")
    monkeypatch.setattr(owner, "_MAX_UINT64", 3)
    with pytest.raises(OverflowError, match="uint64"):
        direct.add_bytes("too-large.plan", b"four")
    assert list(tmp_path.iterdir())[0].stat().st_size == 16 + 65536
    direct.abort()


def test_direct_writer_can_publish_without_sections_and_never_mutates_a_finished_bundle(tmp_path):
    destination = tmp_path / "model.bundle"
    direct = writer(destination)
    direct.finish()
    assert read(destination)[0]["sections"] == {} and read(destination)[1] == b""
    before = destination.read_bytes()
    direct.abort()
    with pytest.raises(RuntimeError, match="finished"):
        direct.add_bytes("late.plan", b"late")
    assert destination.read_bytes() == before


def test_direct_staging_preserves_core_provenance_and_reserved_names(monkeypatch, tmp_path):
    destination = tmp_path / "model.bundle"
    direct = writer(destination)
    request = object()
    observed = []

    def collect(value):
        observed.append(value)
        return {"fixture": "CPU-only provenance", "version": 1}

    monkeypatch.setattr(owner.provenance, "collect", collect)
    direct.record_build(request)
    with pytest.raises(ValueError, match="reserved"):
        direct.add_bytes("trtmc.family.json", b"not allowed")
    direct.add_bytes("engine.plan", b"PLAN")
    direct.finish()
    header, payload, _, _ = read(destination)
    section = header["sections"][owner.provenance.PROVENANCE_SECTION]
    assert observed == [request] and section["offset"] == 4
    assert json.loads(payload[section["offset"] :]) == {
        "fixture": "CPU-only provenance",
        "version": 1,
    }


def test_direct_io_retries_short_writes_in_prefix_sections_and_final_header(monkeypatch, tmp_path):
    original = owner.os.fdopen

    class ShortOutput:
        def __init__(self, output):
            self.output = output

        def __getattr__(self, name):
            return getattr(self.output, name)

        def write(self, value):
            return self.output.write(value[:3])

    monkeypatch.setattr(
        owner.os, "fdopen", lambda *args, **kwargs: ShortOutput(original(*args, **kwargs))
    )
    destination = tmp_path / "model.bundle"
    direct = writer(destination)
    direct.add_bytes("engine.plan", b"every original plan byte")
    direct.finish()
    assert read(destination)[1] == b"every original plan byte"


def test_failed_prefix_write_preserves_destination_and_cleans_only_new_spool(monkeypatch, tmp_path):
    original = owner.os.fdopen

    class FailedOutput:
        def __init__(self, output):
            self.output = output

        def __getattr__(self, name):
            return getattr(self.output, name)

        def write(self, _value):
            return 0

    monkeypatch.setattr(
        owner.os, "fdopen", lambda *args, **kwargs: FailedOutput(original(*args, **kwargs))
    )
    destination = tmp_path / "model.bundle"
    destination.write_bytes(b"previous")
    direct = writer(destination)
    with pytest.raises(OSError, match="no progress"):
        direct.add_bytes("engine.plan", b"new")
    assert destination.read_bytes() == b"previous" and list(tmp_path.iterdir()) == [destination]
    with pytest.raises(RuntimeError, match="write failed"):
        direct.finish()
    direct.abort()
