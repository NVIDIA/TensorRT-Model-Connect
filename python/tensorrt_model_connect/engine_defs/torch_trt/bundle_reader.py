"""Read .trtfb bundle files — for inspect and debug workflows.

Reads the JSON header and optionally extracts binary sections.
Supports both TRTFB (new, raw TRT) and TTRTB (legacy, TorchScript) magic.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .bundle_writer import BUNDLE_MAGIC

# Legacy magic for old .ttrtb bundles (TorchScript-based)
_LEGACY_MAGIC = b"TTRTB\x00\x01\x00"

HEADER_OFFSET = 16  # 8 magic + 8 length


def _check_magic(magic: bytes) -> bool:
    """Check if magic bytes match TRTFB or legacy TTRTB."""
    return magic == BUNDLE_MAGIC or magic == _LEGACY_MAGIC


def has_ttrtb_magic(path: str | Path) -> bool:
    """Check if a file starts with a recognized bundle magic."""
    try:
        with open(path, "rb") as f:
            return _check_magic(f.read(8))
    except (OSError, IOError):
        return False


def read_bundle_header(path: str | Path) -> dict[str, Any]:
    """Read the JSON header from a bundle file.

    Returns the header as a Python dict. Does not load binary sections.
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if not _check_magic(magic):
            raise ValueError(
                f"Not a valid bundle: expected magic {BUNDLE_MAGIC!r} or "
                f"{_LEGACY_MAGIC!r}, got {magic!r}")

        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)

    return json.loads(header_bytes.decode("utf-8"))


def read_bundle_section(path: str | Path, section_name: str) -> bytes:
    """Read a single binary section from a bundle by name.

    Raises KeyError if the section is not found.
    """
    header = read_bundle_header(path)
    sections = header.get("sections", {})

    if section_name not in sections:
        available = list(sections.keys())
        raise KeyError(
            f"Section {section_name!r} not found. Available: {available}")

    section_info = sections[section_name]
    data_start = HEADER_OFFSET + len(
        json.dumps(header, indent=2).encode("utf-8"))

    with open(path, "rb") as f:
        f.seek(data_start + section_info["offset"])
        return f.read(section_info["size"])
