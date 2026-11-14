# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the AIPerf Boltz-2 inputs live in the reference environment (reference/prepare.py's package root and one
preparation cache), and the processed request behind a prepared one, which the native reference featurizes."""

from __future__ import annotations

from pathlib import Path
import struct
import sys

from families.boltz2.model_config import PROCESSED, REQUEST
from families.boltz2.request_preparation import MAGIC, VERSION, prepare_structure_request

PACKAGE = Path(sys.prefix) / "trtmc-reference" / "boltz2-model"
CACHE = Path(sys.prefix) / "trtmc-reference" / "boltz2-requests"


def request_yaml(document: bytes) -> bytes:
    """The request YAML of a prepared request (its first section), or the document itself when it is YAML."""
    if not document.startswith(MAGIC):
        return document
    version, first, *_ = struct.unpack_from("<I4Q", document, len(MAGIC))
    if version != VERSION:
        raise ValueError(f"Boltz-2 prepared request version {version}, expected {VERSION}")
    start = len(MAGIC) + struct.calcsize("<I4Q")
    return document[start:start + first]


def prepare(request: bytes, directory: Path) -> tuple[Path, Path]:
    """Prepare a single-sequence request (no MSA file) through the shared cache: the prepared request and the
    processed directory its features come from."""
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / "request.yaml"
    staged.write_bytes(request)
    result = prepare_structure_request(PACKAGE, staged, directory / "request.b2rq", cache_dir=CACHE)
    key = str(result["cache_key"])
    return directory / "request.b2rq", CACHE / key[:2] / key / "work" / "processed"


def processed_directory(document: bytes, scratch: Path) -> Path:
    """The processed request a document stands for: the package's own for its embedded request, else the cached
    preparation of the document's request YAML."""
    request = request_yaml(document)
    if request == (PACKAGE / REQUEST).read_bytes():
        return PACKAGE / PROCESSED
    return prepare(request, scratch)[1]
