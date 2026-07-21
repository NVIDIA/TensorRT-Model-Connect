# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable family-owned entry point for the Mixtral debug runtime."""

import json
import struct

from .model.runtime import (
    TrtRunner,
    runner_from_bundle,
)


def _read_section(bundle_path: str, section_name: str) -> tuple[bytes | None, dict]:
    with open(bundle_path, "rb") as bundle:
        if bundle.read(8) != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", bundle.read(8))[0]
        header = json.loads(bundle.read(header_len).decode("utf-8"))
        metadata = header.get("sections", {}).get(section_name)
        if metadata is None:
            return None, header
        bundle.seek(16 + header_len + metadata["offset"])
        return bundle.read(metadata["size"]), header


def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named raw section from a Mixtral bundle."""
    return _read_section(bundle_path, section_name)[0]


def load_engine_from_bundle(
    bundle_path: str, section_name: str = "engine_plan"
) -> tuple[bytes, dict]:
    """Load an engine plan and bundle metadata."""
    engine_plan, header = _read_section(bundle_path, section_name)
    if engine_plan is None:
        raise KeyError(f"Bundle {bundle_path!r} does not contain section {section_name!r}")
    return engine_plan, header


def load_config_from_bundle(bundle_path: str) -> dict:
    """Load the embedded Mixtral config, when present."""
    data = load_section_from_bundle(bundle_path, "config.json")
    return {} if data is None else json.loads(data.decode("utf-8"))


__all__ = [
    "TrtRunner",
    "load_config_from_bundle",
    "load_engine_from_bundle",
    "load_section_from_bundle",
    "runner_from_bundle",
]
