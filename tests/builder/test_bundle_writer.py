"""Tests for bundle_writer.py — .trtfb binary format round-trip.

Pure Python, no TRT needed.

Trace: ARCH-BDL-001, UD-BDL-01
Intent: Validate that .trtfb bundles are written with correct magic, header, and section layout and can be read back faithfully.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: Bundle magic bytes, header JSON, section offsets/sizes, and payload data survive a write-then-read round-trip.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.bundle_writer import (
    BUNDLE_MAGIC,
    BundleInfo,
    BundleSection,
    write_bundle,
)

from tests.builder.conftest import read_trtfb_bundle


class TestBundleMagic:
    def test_magic_bytes(self):
        assert BUNDLE_MAGIC == b"TRTFB\x00\x01\x00"
        assert len(BUNDLE_MAGIC) == 8


class TestWriteBundle:
    def _read_bundle(self, path: str) -> tuple[dict, dict[str, bytes]]:
        return read_trtfb_bundle(path)

    def test_single_section(self, tmp_path):
        info = BundleInfo(
            model_id="test-model",
            model_type="qwen3",
            family="qwen",
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            max_cache_length=256,
        )
        data = b"fake engine plan data here"
        sections = [BundleSection("engine_plan", data)]

        out_path = str(tmp_path / "test.trtfb")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert header["model_id"] == "test-model"
        assert header["model_type"] == "qwen3"
        assert header["family"] == "qwen"
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

        out_path = str(tmp_path / "multi.trtfb")
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

        out_path = str(tmp_path / "offsets.trtfb")
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
        out_path = str(tmp_path / "empty.trtfb")
        write_bundle(out_path, info, sections)

        header, sdata = self._read_bundle(out_path)
        assert sdata["empty_sec"] == b""
        assert header["sections"]["empty_sec"]["size"] == 0

    def test_no_sections(self, tmp_path):
        info = BundleInfo(model_id="nosec")
        out_path = str(tmp_path / "nosec.trtfb")
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
        out_path = str(tmp_path / "binary.trtfb")
        write_bundle(out_path, info, sections)

        _, sdata = self._read_bundle(out_path)
        assert sdata["engine_plan"] == binary_data

    def test_all_info_fields(self, tmp_path):
        info = BundleInfo(
            model_id="full-test",
            model_type="llama",
            family="llama",
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
        out_path = str(tmp_path / "full.trtfb")
        write_bundle(out_path, info, [BundleSection("engine_plan", b"x")])

        header, _ = self._read_bundle(out_path)
        assert header["trt_version"] == "10.1.0"
        assert header["trt_abi"] == "10.1"
        assert header["gpu_name"] == "NVIDIA RTX 4090"
        assert header["created_at"] == "2026-02-16T12:00:00Z"
        assert header["num_attention_heads"] == 32
        assert header["num_key_value_heads"] == 8


def _read_bundle_from_bytes(data: bytes) -> tuple[dict, dict[str, bytes]]:
    """Parse a .trtfb bundle from raw bytes. Raises on any format error."""
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
            "model_type": "qwen3",
            "family": "qwen",
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
            _read_bundle_from_bytes(b"TRTFB")

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
        assert header["family"] == "qwen"
        assert len(sections["engine_plan"]) == 16


def test_max_batch_size_roundtrip_and_back_compat(tmp_path):
    """``max_batch_size`` is an additive optional bundle field (Decision C):
    new bundles serialize it, legacy bundles omit it.
    """
    new = BundleInfo(
        model_id="flux", family="diffusion_flux",
        max_batch_size={"dit": 4, "text_encoder": 8, "vae": 1},
    )
    new_path = str(tmp_path / "new.trtfb")
    write_bundle(new_path, new, [BundleSection("engine_plan", b"x")])
    new_header, _ = read_trtfb_bundle(new_path)
    assert new_header["max_batch_size"] == {"dit": 4, "text_encoder": 8, "vae": 1}

    legacy = BundleInfo(model_id="legacy", family="qwen")
    legacy_path = str(tmp_path / "legacy.trtfb")
    write_bundle(legacy_path, legacy, [BundleSection("engine_plan", b"x")])
    legacy_header, _ = read_trtfb_bundle(legacy_path)
    assert "max_batch_size" not in legacy_header
