# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import provenance
from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, BundleWriter

build_core = importlib.import_module("tensorrt_model_connect.build")
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_SOURCE = "89abcdef0123456789abcdef0123456789abcdef"


def _sections(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    header_size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    start = len(BUNDLE_MAGIC) + 8 + header_size
    header = json.loads(data[len(BUNDLE_MAGIC) + 8 : start])
    return {
        name: data[start + entry["offset"] : start + entry["offset"] + entry["length"]]
        for name, entry in header["sections"].items()
    }


def _snapshot(root: Path) -> Path:
    snapshot = root / "hub/models--example-org--example-model/snapshots" / _REVISION
    snapshot.mkdir(parents=True)
    return snapshot


@pytest.fixture
def known_host(monkeypatch):
    monkeypatch.setattr(provenance, "source_state", lambda _package: (_SOURCE, False))
    monkeypatch.setattr(provenance.importlib.metadata, "version", lambda _name: "0.1.0")
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.15.2"))
    monkeypatch.setattr(provenance, "_device_sm", lambda: 100)
    monkeypatch.setattr(provenance.platform, "machine", lambda: "x86_64")


def test_recorded_build_writes_one_core_provenance_section(tmp_path: Path, known_host) -> None:
    destination = tmp_path / "model.bundle"
    request = SimpleNamespace(
        model_dir=_snapshot(tmp_path),
        output_path=destination,
        precision="fp16",
        max_sequence_length=None,
        tensor_parallel_size=2,
        fp32_layers=(1, 3),
        graph_transform=None,
        verbose=True,
    )
    writer = BundleWriter(destination)
    writer.set_header(family="example", task="text_generation", backend="trt")
    writer.record_build(request)
    writer.add_json("runtime.json", {"layers": 2})
    writer.finish()

    sections = _sections(destination)
    assert list(sections) == ["runtime.json", provenance.PROVENANCE_SECTION]
    record = json.loads(sections[provenance.PROVENANCE_SECTION])
    assert record == {
        "version": 1,
        "producer": {
            "name": "tensorrt-model-connect",
            "version": "0.1.0",
            "source_revision": _SOURCE,
            "source_dirty": False,
        },
        "checkpoint": {"hf_id": "example-org/example-model", "revision": _REVISION},
        "build": {
            "precision": "fp16",
            "max_sequence_length": None,
            "tensor_parallel_size": 2,
            "fp32_layers": [1, 3],
            "graph_transform": False,
        },
        "toolchain": {"tensorrt_version": "10.15.2", "sm": 100, "cpu_arch": "x86_64"},
    }
    assert str(tmp_path) not in sections[provenance.PROVENANCE_SECTION].decode()


def test_provenance_is_best_effort_and_never_guesses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(provenance, "source_state", lambda _package: (None, None))
    monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(provenance, "_device_sm", lambda: pytest.fail("CUDA queried"))
    local_checkpoint = tmp_path / "prepared"
    local_checkpoint.mkdir()

    record = provenance.collect(SimpleNamespace(model_dir=local_checkpoint, precision="bf16"))

    assert record["checkpoint"] is None
    assert record["producer"]["source_revision"] is None
    assert record["producer"]["source_dirty"] is None
    assert record["build"] == {"precision": "bf16"}
    assert record["toolchain"]["tensorrt_version"] is None
    assert record["toolchain"]["sm"] is None


def test_checkpoint_identity_requires_an_exact_hugging_face_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(snapshot, target_is_directory=True)
    named = tmp_path / "hub/models--example-org--example-model/snapshots/main"
    named.mkdir()

    expected = {"hf_id": "example-org/example-model", "revision": _REVISION}
    assert provenance.checkpoint_identity(snapshot) == expected
    assert provenance.checkpoint_identity(alias) == expected
    assert provenance.checkpoint_identity(named) is None
    assert provenance.checkpoint_identity(tmp_path) is None
    assert provenance.checkpoint_identity(None) is None


def test_family_sections_cannot_use_the_core_prefix(tmp_path: Path) -> None:
    writer = BundleWriter(tmp_path / "model.bundle")
    with pytest.raises(ValueError, match="reserved for the core"):
        writer.add_json(provenance.PROVENANCE_SECTION, {})
    with pytest.raises(ValueError, match="reserved for the core"):
        with writer.open_section("trtmc.family.bin"):
            pass
    writer.abort()


@pytest.mark.parametrize(
    ("status", "expected"),
    [("", (_SOURCE, False)), (" M core/build.py\n", (_SOURCE, True)), (None, (_SOURCE, None))],
)
def test_source_state_reports_this_checkout(
    monkeypatch, tmp_path: Path, status: str | None, expected: tuple
) -> None:
    package = tmp_path / "repo/core/builder/tensorrt_model_connect"
    package.mkdir(parents=True)

    def git(_directory: Path, *arguments: str) -> str | None:
        if arguments[0] == "rev-parse":
            return f"{tmp_path / 'repo'}\n{_SOURCE}\n"
        return status

    monkeypatch.setattr(provenance, "_git", git)
    assert provenance.source_state(package.resolve()) == expected


def test_source_state_ignores_an_unrelated_enclosing_checkout(monkeypatch, tmp_path: Path) -> None:
    # An installed package inside another project's virtual environment.
    package = tmp_path / "project/.venv/site-packages/tensorrt_model_connect"
    package.mkdir(parents=True)
    monkeypatch.setattr(
        provenance, "_git", lambda _directory, *_arguments: f"{tmp_path / 'project'}\n{_SOURCE}\n"
    )
    assert provenance.source_state(package.resolve()) == (None, None)


def test_core_build_records_the_request(monkeypatch, tmp_path: Path, known_host) -> None:
    def family_build(_request: BuildRequest, writer: BundleWriter) -> None:
        writer.set_header(family="example", task="text_generation", backend="trt")
        writer.add_bytes("engine.plan", b"plan")

    monkeypatch.setattr(build_core, "_select_backend", lambda _backend: None)
    monkeypatch.setattr(
        build_core, "_load_family", lambda _family: SimpleNamespace(build=family_build)
    )
    request = BuildRequest(
        model_dir=_snapshot(tmp_path),
        output_path=tmp_path / "model.bundle",
        family="example",
        task="text_generation",
        precision="fp16",
        quantization="fp8",
    )

    build_core.build(request)

    record = json.loads(_sections(request.output_path)[provenance.PROVENANCE_SECTION])
    assert record["checkpoint"] == {"hf_id": "example-org/example-model", "revision": _REVISION}
    assert record["build"] == {
        "precision": "fp16",
        "max_sequence_length": None,
        "max_batch_size": 1,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "quantization": "fp8",
        "fp32_layers": [],
        "dynamic_kv_cache": False,
        "mtp_seq_len": None,
        "graph_transform": False,
    }
