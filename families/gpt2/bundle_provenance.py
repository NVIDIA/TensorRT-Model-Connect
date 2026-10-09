# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned checkpoint identity for reusing text-generation bundles."""
from __future__ import annotations

import json
import re
from pathlib import Path
import struct


def checkpoint_identity(model_dir: Path) -> dict[str, str] | None:
    """Record the actual immutable HF cache snapshot consumed by the builder."""
    path = model_dir.resolve()
    repository = path.parent.parent.name
    if (path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", path.name)
            or not repository.startswith("models--")):
        return None
    return {"hf_id": repository.removeprefix("models--").replace("--", "/"),
            "revision": path.name}


def _json_object(payload: bytes, description: str) -> dict:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AssertionError(f"invalid prebuilt {description} JSON") from error
    if not isinstance(value, dict):
        raise AssertionError(f"invalid prebuilt {description} object")
    return value


def _read_provenance(bundle: Path) -> dict:
    with bundle.open("rb") as source:
        if source.read(8) != b"BUNDLE\x01\x00":
            raise AssertionError("invalid prebuilt bundle signature")
        size_bytes = source.read(8)
        if len(size_bytes) != 8:
            raise AssertionError("truncated prebuilt bundle header")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < size <= 100 * 1024 * 1024:
            raise AssertionError("invalid prebuilt bundle header size")
        header_bytes = source.read(size)
        if len(header_bytes) != size:
            raise AssertionError("truncated prebuilt bundle header")
        header = _json_object(header_bytes, "header")
        if header.get("family") != "gpt2" or header.get("task") != "text_generation":
            raise AssertionError("prebuilt bundle has the wrong family or task")
        sections = header.get("sections")
        if not isinstance(sections, dict):
            raise AssertionError("invalid prebuilt bundle sections object")
        section = sections.get("checkpoint_provenance.json")
        if section is None:
            raise AssertionError("prebuilt bundle lacks build provenance; rebuild it with TRTMC")
        if not isinstance(section, dict):
            raise AssertionError("invalid prebuilt provenance section object")
        offset, length = section.get("offset"), section.get("length")
        if not (type(offset) is int and type(length) is int and offset >= 0
                and 0 < length <= 1024 * 1024):
            raise AssertionError("invalid prebuilt provenance section range")
        start = 16 + size + offset
        if start + length > bundle.stat().st_size:
            raise AssertionError("prebuilt provenance section exceeds bundle")
        source.seek(start)
        return _json_object(source.read(length), "provenance")


def validate_prebuilt_bundle(
    bundle: Path, model_dir: Path, manifest: dict, revision: str
) -> dict:
    """Compare builder-recorded inputs with the selected, pinned HF snapshot."""
    provenance = _read_provenance(bundle)
    if provenance.get("version") != 1:
        raise AssertionError("unsupported prebuilt provenance version")
    selected = {"hf_id": manifest["hf_id"], "revision": revision}
    if checkpoint_identity(model_dir) != selected:
        raise AssertionError("prebuilt validation requires the selected immutable HF snapshot")
    if provenance.get("checkpoint") != selected:
        raise AssertionError("prebuilt checkpoint provenance differs from the selected checkpoint")
    expected = {
        "precision": manifest["precision"],
        "max_sequence_length": manifest["max_sequence_length"],
        "tensor_parallel_size": manifest["tensor_parallel_size"],
        "quantization": manifest.get("quantization") or "none",
        "fp32_layers": list(manifest.get("fp32_layers", ())),
    }
    if provenance.get("build") != expected:
        raise AssertionError("prebuilt build profile differs from the family manifest")
    return provenance
