#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that every family model has at least one E2E model manifest.

Runs as a CI gate without importing model modules (no TRT/torch required).

Exit codes:
  0  — all family models are covered
  1  — at least one family model has no E2E manifest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"


def iter_manifest_paths() -> list[Path]:
    """Return model-owned E2E manifests."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(MODELS_DIR.glob("*/tests/manifests/*.json"))


def discover_family_names() -> set[str]:
    """Return every directory that declares a family manifest."""
    return {
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir()
        and (path / "MODEL.toml").is_file()
    }


def discover_manifest_families() -> dict[str, list[str]]:
    """Load all E2E manifests and return {family: [model_names]}."""
    families: dict[str, list[str]] = {}
    for manifest_path in iter_manifest_paths():
        with open(manifest_path) as f:
            data = json.load(f)
        family = data.get("family")
        name = data.get("name", manifest_path.stem)
        if family:
            families.setdefault(family, []).append(name)
    return families


def main() -> int:
    family_names = discover_family_names()
    missing_model_entries = {
        name for name in family_names if not (MODELS_DIR / name / "model.py").is_file()
    }
    manifest_families = discover_manifest_families()
    covered = family_names & set(manifest_families)
    uncovered = family_names - set(manifest_families)

    print("=== Family Model E2E Coverage Report ===")
    print(f"Total families declared:        {len(family_names)}")
    print(f"Missing model.py entrypoint: {len(missing_model_entries)}")
    print(f"Covered by E2E manifest: {len(covered)}")
    print(f"Uncovered:               {len(uncovered)}")
    print()

    # Detailed listing
    for name in sorted(family_names):
        models = manifest_families.get(name, [])
        if models:
            print(f"  [OK]      {name} ({len(models)} model(s): {', '.join(models)})")
        else:
            print(f"  [MISSING] {name}")

    print()

    if missing_model_entries:
        print(
            "ERROR: family manifests without model.py: "
            f"{', '.join(sorted(missing_model_entries))}"
        )
        return 1

    if uncovered:
        print(
            f"ERROR: {len(uncovered)} family model(s) have no E2E manifest: "
            f"{', '.join(sorted(uncovered))}"
        )
        print(
            "Add a JSON manifest in "
            "python/tensorrt_model_connect/models/<family>/tests/manifests/ "
            "with 'family' matching the family directory."
        )
        return 1

    print("All family models have E2E manifest coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
