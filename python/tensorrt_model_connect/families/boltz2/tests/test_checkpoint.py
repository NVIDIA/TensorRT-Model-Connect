# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest

from tensorrt_model_connect.families.boltz2.checkpoint import (
    PINNED_PAIRFORMER,
    _load_checkpoint,
    resolve_pairformer_config,
    validate_artifact,
)
from tensorrt_model_connect.families.boltz2.provenance import PinnedArtifact


def _hparams(**pairformer_overrides):
    pairformer = {
        "num_blocks": 64,
        "num_heads": 16,
        "pairwise_head_width": 32,
        "pairwise_num_heads": 4,
        "post_layer_norm": False,
        **pairformer_overrides,
    }
    return {"token_s": 384, "token_z": 128, "pairformer_args": pairformer}


def test_checkpoint_legacy_metadata_uses_pinned_v2_construction_override() -> None:
    assert resolve_pairformer_config(_hparams()) == PINNED_PAIRFORMER


def test_checkpoint_rejects_explicit_non_v2_pairformer() -> None:
    with pytest.raises(ValueError, match="qualified topology"):
        resolve_pairformer_config(_hparams(v2=False))


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_blocks", 63), ("num_heads", 8), ("pairwise_num_heads", 2)],
)
def test_checkpoint_rejects_unqualified_pairformer_topology(field, value) -> None:
    with pytest.raises(ValueError, match="qualified topology"):
        resolve_pairformer_config(_hparams(**{field: value}))


def test_pinned_artifact_checks_size_and_digest(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"pinned")
    expected = PinnedArtifact(
        filename=path.name,
        sha256=hashlib.sha256(b"pinned").hexdigest(),
        size_bytes=6,
    )
    validate_artifact(path, expected)
    path.write_bytes(b"pinnee")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_artifact(path, expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_artifact(path, expected)


def test_checkpoint_cannot_skip_verification_before_process_validation(tmp_path) -> None:
    path = tmp_path / "untrusted.ckpt"
    path.write_bytes(b"not a pinned checkpoint")

    with pytest.raises(ValueError, match="validated in this process"):
        _load_checkpoint(path, verify=False)
