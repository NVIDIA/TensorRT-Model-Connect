# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS debug-runtime entrypoint and bundle readers."""

from __future__ import annotations

import json
import struct

from .model.runtime import TrtRunner, runner_from_bundle


def _bundle_header(bundle_path: str) -> tuple[dict, int]:
    with open(bundle_path, "rb") as bundle:
        if bundle.read(8) != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_size = struct.unpack("<Q", bundle.read(8))[0]
        return json.loads(bundle.read(header_size).decode("utf-8")), header_size


def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Read one GPT-OSS bundle section."""
    header, header_size = _bundle_header(bundle_path)
    metadata = header.get("sections", {}).get(section_name)
    if metadata is None:
        return None
    with open(bundle_path, "rb") as bundle:
        bundle.seek(16 + header_size + metadata["offset"])
        return bundle.read(metadata["size"])


def load_engine_from_bundle(
    bundle_path: str,
    section_name: str = "engine_plan",
) -> tuple[bytes, dict]:
    """Read a GPT-OSS engine and its bundle header."""
    header, _ = _bundle_header(bundle_path)
    engine = load_section_from_bundle(bundle_path, section_name)
    if engine is None:
        raise KeyError(
            f"Bundle {bundle_path!r} does not contain section {section_name!r}")
    return engine, header


def load_config_from_bundle(bundle_path: str) -> dict:
    """Read GPT-OSS config.json from a bundle."""
    config = load_section_from_bundle(bundle_path, "config.json")
    return {} if config is None else json.loads(config.decode("utf-8"))


__all__ = [
    "TrtRunner",
    "load_config_from_bundle",
    "load_engine_from_bundle",
    "load_section_from_bundle",
    "runner_from_bundle",
]
