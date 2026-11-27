# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pinned checkpoint snapshot the family tests read: the cache only, never the network."""

from __future__ import annotations

import json
import os
from pathlib import Path

MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"
MODEL_DIR_ENV = "TRTMC_GRANITE_SPEECH5_MODEL_DIR"


def pinned_checkpoint() -> tuple[str, str]:
    """The one ``(hf_id, hf_revision)`` every manifest of the family pins."""
    pins = set()
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        pins.add((manifest["hf_id"], manifest["hf_revision"]))
    assert len(pins) == 1, f"the family's manifests pin {sorted(pins)}, not one checkpoint"
    return pins.pop()


def model_dir(hf_id: str | None = None, revision: str | None = None) -> Path:
    """The snapshot from ``TRTMC_GRANITE_SPEECH5_MODEL_DIR`` or the local Hugging Face cache."""
    explicit = os.environ.get(MODEL_DIR_ENV)
    if explicit:
        path = Path(explicit)
        assert path.is_dir(), f"{MODEL_DIR_ENV} does not name a directory: {path}"
        return path
    from huggingface_hub import snapshot_download

    if hf_id is None:
        hf_id, revision = pinned_checkpoint()
    try:
        return Path(snapshot_download(repo_id=hf_id, revision=revision, local_files_only=True))
    except Exception as error:  # noqa: BLE001 - every cache miss is the same failure
        raise AssertionError(f"granite_speech5 tests need the cached checkpoint {hf_id} @ {revision}") from error
