# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned config adapter for the exact supplied SAM2 package."""

from __future__ import annotations

from pathlib import Path

from . import archive_contract
from .archive_contract import Sam2ArchiveContractError, Sam2ArchiveDescription


def _require_reference_declarations(description: Sam2ArchiveDescription) -> None:
    provenance = description.provenance
    manifest_sha256 = archive_contract.sha256_file(
        description.root / archive_contract.SHA256SUMS_RELATIVE_PATH
    )
    if manifest_sha256 != archive_contract.REFERENCE_SHA256SUMS_SHA256:
        raise Sam2ArchiveContractError(
            "SAM2 package does not match the supplied reference provenance: SHA256SUMS"
        )
    if provenance.get("declared_matches_reference_config") is not True:
        raise Sam2ArchiveContractError(
            "SAM2 SHA256SUMS does not declare the reference config SHA-256"
        )
    if provenance.get("declared_matches_reference_checkpoint") is not True:
        raise Sam2ArchiveContractError(
            "SAM2 SHA256SUMS does not declare the reference checkpoint SHA-256"
        )


def require_reference_archive(model_dir: str | Path) -> Sam2ArchiveDescription:
    """Authenticate the complete immutable input package or fail closed."""

    description = archive_contract.describe_archive(model_dir, verify_provenance=True)
    _require_reference_declarations(description)
    provenance = description.provenance
    required_matches = {
        "SHA256SUMS": provenance.get("matches_reference_manifest"),
        archive_contract.CONFIG_RELATIVE_PATH.as_posix(): provenance.get(
            "matches_reference_config"
        ),
        archive_contract.CHECKPOINT_RELATIVE_PATH.as_posix(): provenance.get(
            "matches_reference_checkpoint"
        ),
    }
    mismatches = [name for name, matches in required_matches.items() if matches is not True]
    if mismatches:
        raise Sam2ArchiveContractError(
            "SAM2 package does not match the supplied reference provenance: "
            + ", ".join(mismatches)
        )
    return description


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return a synthetic config only for the authenticated SAM2 delivery.

    A directory without the SAM2 layout is not ours. Once all three contract
    paths are present, malformed or adjacent assets are errors rather than a
    reason to fall through to another family adapter.
    """

    if archive_contract.resolve_package_root(model_dir) is None:
        return None
    # Discovery authenticates the small manifest and its exact declarations,
    # but deliberately avoids hashing the 194 MB checkpoint. The build hook
    # performs full byte-level verification again immediately before exec.
    description = archive_contract.describe_archive(model_dir, verify_provenance=False)
    _require_reference_declarations(description)
    model = description.config["model"]
    memory_attention = model["memory_attention"]
    return {
        "model_type": "sam2",
        "architectures": ["Sam2BBoxVideoTracking"],
        "hidden_size": int(memory_attention["d_model"]),
        "intermediate_size": int(memory_attention["layer"]["dim_feedforward"]),
        "num_hidden_layers": int(memory_attention["num_layers"]),
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": int(model["image_size"]),
        "sam2_model_id": "sam2.1-hiera-small-bbox",
        "sam2_runtime_strategy": "sam2_bbox_video_tracking",
        "sam2_precision": "mixed_bf16_fp32",
        "sam2_qualification": "unqualified",
        "sam2_runtime_eligible": False,
    }


__all__ = ["config_from_dir", "require_reference_archive"]
