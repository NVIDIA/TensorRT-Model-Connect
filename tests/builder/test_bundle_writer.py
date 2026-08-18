# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bundle_writer.py — .bundle binary format round-trip.

Pure Python, no TRT needed.

Trace: ARCH-BDL-001, UD-BDL-01
Intent: Validate that .bundle artifacts are written with correct magic, header, and section layout and can be read back faithfully.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: Bundle magic bytes, header JSON, section offsets/sizes, and payload data survive a write-then-read round-trip.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import struct

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.bundle_writer import (
    BUNDLE_MAGIC,
    BundleInfo,
    BundleSection,
    FileBundleSection,
    _bundle_section_from_file,
    bundle_section_from_file,
    write_bundle,
)

from tests.builder.conftest import read_bundle_file


class TestBundleMagic:
    def test_magic_bytes(self):
        assert BUNDLE_MAGIC == b"BUNDLE\x01\x00"
        assert len(BUNDLE_MAGIC) == 8


def test_bundle_section_public_constructor_remains_name_and_bytes_only():
    assert tuple(inspect.signature(BundleSection).parameters) == ("name", "data")
    assert not hasattr(BundleSection, "from_file")


def test_file_bundle_section_public_factory_requires_size_and_sha(tmp_path):
    source = tmp_path / "engine.plan"
    source.write_bytes(b"plan")
    section = bundle_section_from_file(
        "engine_plan",
        source,
        expected_size=4,
        expected_sha256=hashlib.sha256(b"plan").hexdigest(),
    )
    assert isinstance(section, FileBundleSection)
    assert section.name == "engine_plan"
    assert section.source_path == source
    assert section.expected_size == 4


class TestWriteBundle:
    def _read_bundle(self, path: str) -> tuple[dict, dict[str, bytes]]:
        return read_bundle_file(path)

    def test_single_section(self, tmp_path):
        info = BundleInfo(
            model_id="test-model",
            model_type="example_decoder",
            family="example_family",
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            max_cache_length=256,
        )
        data = b"fake engine plan data here"
        sections = [BundleSection("engine_plan", data)]

        out_path = str(tmp_path / "test.bundle")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert header["model_id"] == "test-model"
        assert header["model_type"] == "example_decoder"
        assert header["family"] == "example_family"
        assert header["vocab_size"] == 32000
        assert header["hidden_size"] == 1024
        assert header["num_layers"] == 12
        assert header["max_cache_length"] == 256
        assert sdata["engine_plan"] == data

    def test_multi_section(self, tmp_path):
        info = BundleInfo(model_id="multi")
        section1 = BundleSection("engine_plan", b"ENGINE" * 100)
        section2 = BundleSection("config.json", b'{"test": true}')
        section3 = BundleSection("tokenizer.json", b'{"vocab": []}')

        out_path = str(tmp_path / "multi.bundle")
        write_bundle(out_path, info, [section1, section2, section3])

        header, sdata = self._read_bundle(out_path)
        assert len(header["sections"]) == 3
        assert sdata["engine_plan"] == b"ENGINE" * 100
        assert sdata["config.json"] == b'{"test": true}'
        assert sdata["tokenizer.json"] == b'{"vocab": []}'

    def test_section_offsets(self, tmp_path):
        info = BundleInfo(model_id="offsets")
        s1_data = b"A" * 100
        s2_data = b"B" * 200
        s3_data = b"C" * 50
        sections = [
            BundleSection("s1", s1_data),
            BundleSection("s2", s2_data),
            BundleSection("s3", s3_data),
        ]

        out_path = str(tmp_path / "offsets.bundle")
        write_bundle(out_path, info, sections)

        header, _ = self._read_bundle(out_path)
        secs = header["sections"]
        assert secs["s1"]["offset"] == 0
        assert secs["s1"]["size"] == 100
        assert secs["s2"]["offset"] == 100
        assert secs["s2"]["size"] == 200
        assert secs["s3"]["offset"] == 300
        assert secs["s3"]["size"] == 50

    def test_empty_section(self, tmp_path):
        info = BundleInfo(model_id="empty")
        sections = [BundleSection("empty_sec", b"")]
        out_path = str(tmp_path / "empty.bundle")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert sdata["empty_sec"] == b""
        assert header["sections"]["empty_sec"]["size"] == 0

    def test_no_sections(self, tmp_path):
        info = BundleInfo(model_id="nosec")
        out_path = str(tmp_path / "nosec.bundle")
        write_bundle(out_path, info, [])

        header, sdata = self._read_bundle(out_path)
        assert len(header["sections"]) == 0
        assert len(sdata) == 0

    def test_binary_data_integrity(self, tmp_path):
        """Verify binary data survives the round-trip exactly."""
        rng = np.random.RandomState(42)
        binary_data = rng.bytes(4096)

        info = BundleInfo(model_id="binary")
        sections = [BundleSection("engine_plan", binary_data)]
        out_path = str(tmp_path / "binary.bundle")
        write_bundle(out_path, info, sections)

        _, sdata = self._read_bundle(out_path)
        assert sdata["engine_plan"] == binary_data

    def test_file_backed_section_is_streamed(self, tmp_path, monkeypatch):
        source = tmp_path / "llm.engine"
        payload = b"edge-engine" * 131072
        source.write_bytes(payload)
        monkeypatch.setattr(
            type(source),
            "read_bytes",
            lambda _self: (_ for _ in ()).throw(
                AssertionError("file-backed sections must not call read_bytes")
            ),
        )

        out_path = tmp_path / "file-backed.bundle"
        write_bundle(
            out_path,
            BundleInfo(model_id="edge"),
            [_bundle_section_from_file("edge/artifacts/llm.engine", source)],
        )

        header, sections = self._read_bundle(str(out_path))
        assert header["sections"]["edge/artifacts/llm.engine"]["size"] == len(payload)
        assert sections["edge/artifacts/llm.engine"] == payload

    def test_public_file_backed_sections_preserve_order_without_read_bytes(
        self, tmp_path, monkeypatch
    ):
        sources = []
        sections = []
        for index, payload in enumerate((b"first", b"second", b"third")):
            source = tmp_path / f"{index}.plan"
            source.write_bytes(payload)
            sources.append(source)
            sections.append(
                bundle_section_from_file(
                    f"plan_{index}",
                    source,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
        monkeypatch.setattr(
            type(sources[0]),
            "read_bytes",
            lambda _self: (_ for _ in ()).throw(
                AssertionError("file-backed sections must be streamed")
            ),
        )
        destination = tmp_path / "ordered.bundle"
        write_bundle(destination, BundleInfo(model_id="ordered"), sections)
        header, payloads = self._read_bundle(str(destination))
        assert list(header["sections"]) == ["plan_0", "plan_1", "plan_2"]
        assert list(payloads) == ["plan_0", "plan_1", "plan_2"]

    def test_public_file_backed_section_rejects_symlink_size_and_sha(self, tmp_path):
        source = tmp_path / "engine.plan"
        source.write_bytes(b"engine")
        symlink = tmp_path / "link.plan"
        symlink.symlink_to(source.name)
        digest = hashlib.sha256(b"engine").hexdigest()
        with pytest.raises(ValueError, match="not a regular file"):
            bundle_section_from_file(
                "symlink", symlink, expected_size=len(b"engine"), expected_sha256=digest
            )
        with pytest.raises(ValueError, match="source size mismatch"):
            bundle_section_from_file(
                "wrong_size",
                source,
                expected_size=len(b"engine") + 1,
                expected_sha256=digest,
            )
        with pytest.raises(ValueError, match="invalid expected_sha256"):
            bundle_section_from_file(
                "bad_sha", source, expected_size=len(b"engine"), expected_sha256="bad"
            )
        with pytest.raises(ValueError, match="require expected_sha256"):
            bundle_section_from_file(
                "missing_sha", source, expected_size=len(b"engine"), expected_sha256=None
            )

    @pytest.mark.parametrize("invalid_size", [-1, True, "6"])
    def test_public_file_backed_section_rejects_invalid_expected_size(
        self, tmp_path, invalid_size
    ):
        source = tmp_path / "engine.plan"
        source.write_bytes(b"engine")
        with pytest.raises(ValueError, match="invalid expected_size"):
            bundle_section_from_file(
                "bad_size",
                source,
                expected_size=invalid_size,
                expected_sha256=hashlib.sha256(b"engine").hexdigest(),
            )

    def test_duplicate_names_are_rejected_before_overwriting_destination(self, tmp_path):
        destination = tmp_path / "existing.bundle"
        destination.write_bytes(b"keep")
        source = tmp_path / "same.plan"
        source.write_bytes(b"b")
        file_section = bundle_section_from_file(
            "same",
            source,
            expected_size=1,
            expected_sha256=hashlib.sha256(b"b").hexdigest(),
        )
        with pytest.raises(ValueError, match="non-empty and unique"):
            write_bundle(
                destination,
                BundleInfo(model_id="duplicate"),
                [BundleSection("same", b"a"), file_section],
            )
        assert destination.read_bytes() == b"keep"

    def test_file_backed_digest_mismatch_is_not_published(self, tmp_path):
        source = tmp_path / "mutable.engine"
        source.write_bytes(b"engine")
        destination = tmp_path / "must-not-exist.bundle"

        with pytest.raises(RuntimeError, match="changed after validation"):
            write_bundle(
                destination,
                BundleInfo(model_id="edge"),
                [
                    _bundle_section_from_file(
                        "edge/llm.engine",
                        source,
                        expected_sha256="0" * 64,
                    )
                ],
            )

        assert not destination.exists()
        assert not list(tmp_path.glob(f".{destination.name}.tmp.*"))

    def test_all_info_fields(self, tmp_path):
        info = BundleInfo(
            model_id="full-test",
            model_type="example_decoder",
            family="example_family",
            trt_version="10.1.0",
            trt_abi="10.1",
            gpu_name="NVIDIA RTX 4090",
            created_at="2026-02-16T12:00:00Z",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_cache_length=512,
        )
        out_path = str(tmp_path / "full.bundle")
        write_bundle(out_path, info, [BundleSection("engine_plan", b"x")])

        header, _ = self._read_bundle(out_path)
        assert header["trt_version"] == "10.1.0"
        assert header["trt_abi"] == "10.1"
        assert header["gpu_name"] == "NVIDIA RTX 4090"
        assert header["created_at"] == "2026-02-16T12:00:00Z"
        assert header["num_attention_heads"] == 32
        assert header["num_key_value_heads"] == 8


def _read_bundle_from_bytes(data: bytes) -> tuple[dict, dict[str, bytes]]:
    """Parse a .bundle artifact from raw bytes. Raises on any format error."""
    if len(data) < 8:
        raise ValueError("File too short to contain magic bytes")
    magic = data[:8]
    if magic != BUNDLE_MAGIC:
        raise ValueError(
            f"Bad magic bytes: {magic!r}, expected {BUNDLE_MAGIC!r}")
    if len(data) < 16:
        raise ValueError("File too short to contain header length")
    header_len = struct.unpack("<Q", data[8:16])[0]
    if len(data) < 16 + header_len:
        raise ValueError(
            f"File truncated: header declares {header_len} bytes but only "
            f"{len(data) - 16} bytes remain after the 16-byte preamble")
    header = json.loads(data[16:16 + header_len].decode("utf-8"))

    sections_data: dict[str, bytes] = {}
    data_start = 16 + header_len
    for name, meta in header.get("sections", {}).items():
        sec_start = data_start + meta["offset"]
        sec_end = sec_start + meta["size"]
        if sec_end > len(data):
            raise ValueError(
                f"Section {name!r} extends beyond file: needs byte "
                f"{sec_end} but file is {len(data)} bytes")
        sections_data[name] = data[sec_start:sec_end]

    return header, sections_data


class TestCorruptedBundles:
    """Verify that corrupted bundle data raises clear errors during parsing."""

    def _make_valid_bundle_bytes(self) -> bytes:
        """Build a minimal valid bundle in memory and return its bytes."""
        header = {
            "model_id": "corrupt-test",
            "model_type": "example_decoder",
            "family": "example_family",
            "vocab_size": 100,
            "hidden_size": 64,
            "num_layers": 1,
            "sections": {
                "engine_plan": {"offset": 0, "size": 16},
            },
        }
        header_json = json.dumps(header).encode("utf-8")
        section_data = b"ENGINEPLANDATA!!"  # exactly 16 bytes
        buf = bytearray()
        buf += BUNDLE_MAGIC
        buf += struct.pack("<Q", len(header_json))
        buf += header_json
        buf += section_data
        return bytes(buf)

    def test_bad_magic_bytes(self):
        """Wrong magic bytes should raise ValueError."""
        data = self._make_valid_bundle_bytes()
        # Corrupt the first byte of the magic
        corrupted = b"XRTFB\x00\x01\x00" + data[8:]
        with pytest.raises(ValueError, match="Bad magic bytes"):
            _read_bundle_from_bytes(corrupted)

    def test_zeroed_magic_bytes(self):
        """All-zero magic bytes should raise ValueError."""
        data = self._make_valid_bundle_bytes()
        corrupted = b"\x00" * 8 + data[8:]
        with pytest.raises(ValueError, match="Bad magic bytes"):
            _read_bundle_from_bytes(corrupted)

    def test_truncated_before_magic(self):
        """File shorter than 8 bytes cannot contain valid magic."""
        with pytest.raises(ValueError, match="too short to contain magic"):
            _read_bundle_from_bytes(b"BUNDLE")

    def test_empty_file(self):
        """Empty file should raise ValueError."""
        with pytest.raises(ValueError, match="too short to contain magic"):
            _read_bundle_from_bytes(b"")

    def test_truncated_header_length(self):
        """File with valid magic but truncated before header length completes."""
        # Only 8 magic bytes + 4 bytes of header length (need 8)
        truncated = BUNDLE_MAGIC + b"\x10\x00\x00\x00"
        with pytest.raises(ValueError, match="too short to contain header length"):
            _read_bundle_from_bytes(truncated)

    def test_truncated_header_json(self):
        """Header length declares more JSON bytes than available in the file."""
        # Declare a 1000-byte header but only provide 10 bytes
        header_len = struct.pack("<Q", 1000)
        truncated = BUNDLE_MAGIC + header_len + b'{"a": "b"}'
        with pytest.raises(ValueError, match="File truncated.*header declares"):
            _read_bundle_from_bytes(truncated)

    def test_invalid_header_json(self):
        """Valid preamble but invalid JSON in the header region."""
        bad_json = b"this is not valid json {{{{"
        header_len = struct.pack("<Q", len(bad_json))
        data = BUNDLE_MAGIC + header_len + bad_json
        with pytest.raises(json.JSONDecodeError):
            _read_bundle_from_bytes(data)

    def test_truncated_section_data(self):
        """Header declares a section larger than remaining file bytes."""
        header = {
            "model_id": "truncated",
            "sections": {
                "engine_plan": {"offset": 0, "size": 9999},
            },
        }
        header_json = json.dumps(header).encode("utf-8")
        # Only provide 10 bytes of section data (header says 9999)
        data = BUNDLE_MAGIC + struct.pack("<Q", len(header_json)) + header_json + b"x" * 10
        with pytest.raises(ValueError, match="extends beyond file"):
            _read_bundle_from_bytes(data)

    def test_section_offset_past_eof(self):
        """Section offset points beyond the file boundary."""
        header = {
            "model_id": "bad-offset",
            "sections": {
                "engine_plan": {"offset": 50000, "size": 16},
            },
        }
        header_json = json.dumps(header).encode("utf-8")
        data = BUNDLE_MAGIC + struct.pack("<Q", len(header_json)) + header_json + b"x" * 16
        with pytest.raises(ValueError, match="extends beyond file"):
            _read_bundle_from_bytes(data)

    def test_valid_bundle_parses_successfully(self):
        """Sanity check: a valid bundle should parse without error."""
        data = self._make_valid_bundle_bytes()
        header, sections = _read_bundle_from_bytes(data)
        assert header["model_id"] == "corrupt-test"
        assert header["family"] == "example_family"
        assert len(sections["engine_plan"]) == 16


def test_max_batch_size_roundtrip_and_back_compat(tmp_path):
    """``max_batch_size`` is an additive optional bundle field (Decision C):
    new bundles serialize it, legacy bundles omit it.
    """
    new = BundleInfo(
        model_id="batched-model", family="batched_family",
        max_batch_size={"dit": 4, "text_encoder": 8, "vae": 1},
    )
    new_path = str(tmp_path / "new.bundle")
    write_bundle(new_path, new, [BundleSection("engine_plan", b"x")])
    new_header, _ = read_bundle_file(new_path)
    assert new_header["max_batch_size"] == {"dit": 4, "text_encoder": 8, "vae": 1}

    legacy = BundleInfo(model_id="legacy", family="legacy_family")
    legacy_path = str(tmp_path / "legacy.bundle")
    write_bundle(legacy_path, legacy, [BundleSection("engine_plan", b"x")])
    legacy_header, _ = read_bundle_file(legacy_path)
    assert "max_batch_size" not in legacy_header
