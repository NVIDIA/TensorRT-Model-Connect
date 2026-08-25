import json
import struct
import pytest
from tensorrt_model_connect.bundle_writer import BundleReader, BUNDLE_MAGIC

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
            "config": {"offset": 0, "size": 4},
            "weights": {"offset": 4, "size": 8}
        }
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
            "config": {"offset": "0", "size": 4}
        }
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"conf"])
    with pytest.raises(ValueError, match="offset and size must be integers"):
        BundleReader(p)

def test_negative_offset(tmp_path):
    p = tmp_path / "negative_offset.bundle"
    header = {
        "sections": {
            "config": {"offset": -1, "size": 4}
        }
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"conf"])
    with pytest.raises(ValueError, match="offset and size must be non-negative"):
        BundleReader(p)

def test_overlapping_sections(tmp_path):
    p = tmp_path / "overlap.bundle"
    header = {
        "sections": {
            "config": {"offset": 0, "size": 4},
            "weights": {"offset": 2, "size": 4}
        }
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"overlap_"])
    with pytest.raises(ValueError, match="Sections overlap"):
        BundleReader(p)

def test_out_of_file_range(tmp_path):
    p = tmp_path / "out_of_range.bundle"
    header = {
        "sections": {
            "config": {"offset": 0, "size": 100}
        }
    }
    create_bundle(p, BUNDLE_MAGIC, header, [b"short"])
    with pytest.raises(ValueError, match="goes out-of-file range"):
        BundleReader(p)

def test_duplicate_sections(tmp_path):
    p = tmp_path / "duplicate.bundle"
    header_str = '{"sections": {"config": {"offset": 0, "size": 4}, "config": {"offset": 4, "size": 4}}}'.encode("utf-8")
    with open(p, "wb") as f:
        f.write(BUNDLE_MAGIC)
        f.write(struct.pack("<Q", len(header_str)))
        f.write(header_str)
        f.write(b"confconf")
    with pytest.raises(ValueError, match="Duplicate section name found: config"):
        BundleReader(p)
