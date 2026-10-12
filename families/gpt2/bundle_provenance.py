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
    assert isinstance(value, dict), f"invalid prebuilt {description} object"
    return value


def _read_provenance(bundle: Path) -> dict:
    with bundle.open("rb") as source:
        assert source.read(8) == b"BUNDLE\x01\x00", "invalid prebuilt bundle signature"
        size_bytes = source.read(8)
        assert len(size_bytes) == 8, "truncated prebuilt bundle header"
        size = struct.unpack("<Q", size_bytes)[0]
        assert 0 < size <= 100 * 1024 * 1024, "invalid prebuilt bundle header size"
        header_bytes = source.read(size)
        assert len(header_bytes) == size, "truncated prebuilt bundle header"
        header = _json_object(header_bytes, "header")
        assert header.get("family") == "gpt2" and header.get("task") == "text_generation", (
            "prebuilt bundle has the wrong family or task"
        )
        sections = header.get("sections")
        assert isinstance(sections, dict), "invalid prebuilt bundle sections object"
        section = sections.get("checkpoint_provenance.json")
        assert section is not None, "prebuilt bundle lacks build provenance; rebuild it with TRTMC"
        assert isinstance(section, dict), "invalid prebuilt provenance section object"
        offset, length = section.get("offset"), section.get("length")
        assert (type(offset) is int and type(length) is int and offset >= 0
                and 0 < length <= 1024 * 1024), "invalid prebuilt provenance section range"
        start = 16 + size + offset
        assert start + length <= bundle.stat().st_size, "prebuilt provenance section exceeds bundle"
        source.seek(start)
        return _json_object(source.read(length), "provenance")


def validate_prebuilt_bundle(
    bundle: Path, model_dir: Path, manifest: dict, revision: str
) -> dict:
    """Compare builder-recorded inputs with the selected, pinned HF snapshot."""
    provenance = _read_provenance(bundle)
    assert provenance.get("version") == 1, "unsupported prebuilt provenance version"
    selected = {"hf_id": manifest["hf_id"], "revision": revision}
    assert checkpoint_identity(model_dir) == selected, (
        "prebuilt validation requires the selected immutable HF snapshot"
    )
    assert provenance.get("checkpoint") == selected, (
        "prebuilt checkpoint provenance differs from the selected checkpoint"
    )
    expected = {
        "precision": manifest["precision"],
        "max_sequence_length": manifest["max_sequence_length"],
        "tensor_parallel_size": manifest["tensor_parallel_size"],
        "quantization": manifest.get("quantization") or "none",
        "fp32_layers": list(manifest.get("fp32_layers", ())),
    }
    assert provenance.get("build") == expected, "prebuilt build profile differs from the family manifest"
    return provenance
