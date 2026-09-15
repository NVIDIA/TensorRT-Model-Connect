# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, _MAX_HEADER_SIZE


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



def create_bundle(path, magic, header, sections_data):
    header_str = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<Q", len(header_str)))
        f.write(header_str)
        for data in sections_data:
            f.write(data)

def test_valid_bundle(tmp_path):
    p = tmp_path / "valid.bundle"
    header = {
        "model_id": "test",
        "sections": {
            "config": {"offset": 0, "length": 4},
            "weights": {"offset": 4, "length": 8},
        },
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"conf", b"weightss"])
    reader = BundleReader(p)
    assert reader.read_section("config") == b"conf"
    assert reader.read_section("weights") == b"weightss"

def test_invalid_magic(tmp_path):
    p = tmp_path / "invalid_magic.bundle"
    header = {"sections": {}}
    create_bundle(p, b"BADMAGIC", header, [])
    with pytest.raises(ValueError, match="Invalid bundle magic signature"):
        BundleReader(p)


def test_missing_sections(tmp_path):
    p = tmp_path / "missing_sections.bundle"
    header = {"model_id": "test"}
    create_bundle(p, BUNDLE_MAGIC, header, [])
    with pytest.raises(ValueError, match="missing 'sections' key"):
        BundleReader(p)

def test_invalid_offset_type(tmp_path):
    p = tmp_path / "invalid_offset.bundle"
    header = {
        "sections": {
            "config": {"offset": "0", "length": 4},
        },
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"conf"])
    with pytest.raises(ValueError, match="offset and length must be integers"):
        BundleReader(p)


def test_negative_offset(tmp_path):
    p = tmp_path / "negative_offset.bundle"
    header = {
        "sections": {
            "config": {"offset": -1, "length": 4},
        },
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"conf"])
    with pytest.raises(ValueError, match="offset and length must be non-negative"):
        BundleReader(p)


def test_overlapping_sections(tmp_path):
    p = tmp_path / "overlap.bundle"
    header = {
        "sections": {
            "config": {"offset": 0, "length": 4},
            "weights": {"offset": 2, "length": 4},
        },
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"overlap_"])
    with pytest.raises(ValueError, match="Sections overlap"):
        BundleReader(p)


def test_out_of_file_range(tmp_path):
    p = tmp_path / "out_of_range.bundle"
    header = {
        "sections": {
            "config": {"offset": 0, "length": 100},
        },
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"short"])
    with pytest.raises(ValueError, match="goes out-of-file range"):
        BundleReader(p)


def test_duplicate_sections(tmp_path):
    """Bundles with duplicate section names in JSON must be rejected."""
    p = tmp_path / "duplicate.bundle"
    # Craft raw bytes: json.dumps deduplicates keys, so write raw bytes.
    raw_sections = b'"config":{"offset":0,"length":4},"config":{"offset":4,"length":4}'
    header_str = b'{"sections":{' + raw_sections + b"}}"  # deliberately invalid JSON key duplication
    with open(p, "wb") as f:
        f.write(BUNDLE_MAGIC)
        f.write(struct.pack("<Q", len(header_str)))
        f.write(header_str)
        f.write(b"confconf")
    with pytest.raises(ValueError, match="Duplicate section name"):
        BundleReader(p)
