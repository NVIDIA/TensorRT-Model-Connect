#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a pinned OpenPI checkpoint into a reproducible TRTMC build directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .checkpoint_reader import file_sha256, open_checkpoint
from .model_config import (
    OPENPI_MODEL_TYPE,
    OPENPI_UPSTREAM_COMMIT,
    get_profile,
    profile_names,
)
from .tokenizer_export import export_paligemma_bpe_model
from .weight_mapper import MappingRule, map_weights


_MARKER_NAME = ".trtmc-openpi-model-dir"
_MARKER_CONTENT = "trtmc.openpi.prepared.v1\n"


def _validate_regular_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def _load_and_validate_norm_stats(path: Path, *, state_dim: int, action_dim: int) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"normalization statistics are not valid JSON: {path}") from exc
    stats = payload.get("norm_stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        raise ValueError("normalization statistics must contain a norm_stats object")
    for field, minimum_dim in (("state", state_dim), ("actions", action_dim)):
        entry = stats.get(field)
        if not isinstance(entry, dict):
            raise ValueError(f"normalization statistics are missing {field!r}")
        q01 = entry.get("q01")
        q99 = entry.get("q99")
        if not isinstance(q01, list) or not isinstance(q99, list):
            raise ValueError(f"quantile normalization requires q01/q99 arrays for {field!r}")
        if len(q01) < minimum_dim or len(q99) < minimum_dim:
            raise ValueError(
                f"normalization field {field!r} has fewer than {minimum_dim} dimensions"
            )
        for index, (low, high) in enumerate(zip(q01, q99, strict=True)):
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(f"normalization field {field!r}[{index}] is not numeric")
            if index < minimum_dim and high <= low:
                raise ValueError(f"normalization field {field!r}[{index}] has q99 <= q01")
            if index >= minimum_dim and high < low:
                raise ValueError(f"normalization field {field!r}[{index}] has q99 < q01")
    return payload


def _save_weights(path: Path, weights: dict[str, Any]) -> None:
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to write converted OpenPI weights") from exc
    save_file(weights, str(path))


def _safe_replace_output(staging: Path, output: Path, *, force: bool) -> None:
    if output.exists():
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        elif force:
            marker = output / _MARKER_NAME
            if not marker.is_file() or marker.read_text(encoding="utf-8") != _MARKER_CONTENT:
                raise FileExistsError(
                    f"refusing to replace unowned output directory despite --force: {output}"
                )
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"output directory already exists and is not empty: {output}")
    os.replace(staging, output)


def prepare_model_dir(
    args: argparse.Namespace,
    *,
    rules: Sequence[MappingRule] | None = None,
) -> dict[str, Any]:
    """Prepare one profile and return its machine-readable provenance summary.

    ``rules`` is an internal test seam for tiny synthetic checkpoints.  CLI
    callers always use the complete audited mapping returned by the family.
    """

    profile = get_profile(str(args.profile))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    tokenizer = _validate_regular_file(args.tokenizer, "PaliGemma tokenizer")
    norm_stats = _validate_regular_file(args.norm_stats, "normalization statistics")
    _load_and_validate_norm_stats(
        norm_stats,
        state_dim=profile.external_state_dim,
        action_dim=profile.external_action_dim,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    force = bool(getattr(args, "force", False))
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()) and not force:
            raise FileExistsError(f"output directory already exists and is not empty: {output}")

    reader = open_checkpoint(checkpoint)
    mapped = map_weights(reader, profile, rules=rules)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        weights_path = staging / "model.safetensors"
        _save_weights(weights_path, mapped.weights)

        tokenizer_target = staging / "tokenizer.model"
        tokenizer_metadata = export_paligemma_bpe_model(tokenizer, tokenizer_target)
        norm_target = staging / "preprocessor_config.json"
        shutil.copyfile(norm_stats, norm_target)

        manifest = dict(mapped.manifest)
        manifest["artifacts"] = {
            "weights": {
                "file": weights_path.name,
                "sha256": file_sha256(weights_path),
            },
            "tokenizer": {
                "file": tokenizer_target.relative_to(staging).as_posix(),
                "format": "TRTMCBPE",
                "sha256": tokenizer_metadata.asset_sha256,
                **tokenizer_metadata.to_dict(),
            },
            "normalization": {
                "file": norm_target.relative_to(staging).as_posix(),
                "source_sha256": file_sha256(norm_stats),
                "sha256": file_sha256(norm_target),
            },
        }
        manifest_path = staging / "openpi_conversion_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_sha256 = file_sha256(manifest_path)

        config = {
            "model_type": OPENPI_MODEL_TYPE,
            "profile": profile.name,
            "upstream_commit": OPENPI_UPSTREAM_COMMIT,
            "checkpoint_uri": profile.checkpoint_uri,
            "checkpoint_identity_sha256": reader.identity_sha256,
            "conversion_manifest": manifest_path.name,
            "conversion_manifest_sha256": manifest_sha256,
            "weights": weights_path.name,
            "tokenizer": tokenizer_target.relative_to(staging).as_posix(),
            "tokenizer_format": "TRTMCBPE",
            "tokenizer_sha256": tokenizer_metadata.asset_sha256,
            "tokenizer_source_sha256": tokenizer_metadata.source_sha256,
            "tokenizer_export": tokenizer_metadata.to_dict(),
            "normalization": norm_target.relative_to(staging).as_posix(),
            "normalization_sha256": file_sha256(norm_target),
            "policy": profile.to_dict(),
        }
        config_path = staging / "openpi_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / _MARKER_NAME).write_text(_MARKER_CONTENT, encoding="utf-8")

        _safe_replace_output(staging, output, force=force)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "output_dir": str(output),
        "profile": profile.name,
        "upstream_commit": OPENPI_UPSTREAM_COMMIT,
        "checkpoint_identity_sha256": reader.identity_sha256,
        "source_tensor_count": len(mapped.manifest["source_tensors"]),
        "destination_tensor_count": len(mapped.weights),
        "conversion_manifest": str(output / "openpi_conversion_manifest.json"),
        "conversion_manifest_sha256": manifest_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=profile_names())
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="local official checkpoint directory, its params directory, or a synthetic .npz",
    )
    parser.add_argument("--tokenizer", required=True, help="local PaliGemma tokenizer model")
    parser.add_argument("--norm-stats", required=True, help="profile norm_stats.json")
    parser.add_argument("--output", required=True, help="TRTMC model directory to create")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing directory created by this tool",
    )
    result = prepare_model_dir(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
