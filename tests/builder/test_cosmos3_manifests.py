"""Validation tests for the Cosmos 3 E2E manifests.

Ensures the cosmos3-* JSON manifests are valid against the manifest_loader
schema and that key fields (model_type, runtime_strategy, family) match
the plugin's matcher and registered runtime strategy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect.families.cosmos3.plugin import Cosmos3Plugin
from tests.e2e_harness.manifest_loader import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "tests" / "e2e" / "models"


@pytest.fixture(scope="module")
def cosmos3_manifests():
    """Discover all cosmos3-* manifests in the standard E2E directory."""
    return sorted(MANIFEST_DIR.glob("cosmos3-*.json"))


def test_at_least_three_manifests_present(cosmos3_manifests):
    # We ship Super, Super-Reasoner, and Nano lanes.
    names = [m.stem for m in cosmos3_manifests]
    assert "cosmos3-super" in names
    assert "cosmos3-super-reasoner" in names
    assert "cosmos3-nano" in names


@pytest.mark.parametrize("manifest_name", [
    "cosmos3-super",
    "cosmos3-super-reasoner",
    "cosmos3-nano",
])
def test_manifest_loads(manifest_name):
    path = MANIFEST_DIR / f"{manifest_name}.json"
    case = load_manifest(path)
    assert case.name == manifest_name


@pytest.mark.parametrize("manifest_name", [
    "cosmos3-super",
    "cosmos3-super-reasoner",
    "cosmos3-nano",
])
def test_manifest_family_matches_plugin(manifest_name):
    path = MANIFEST_DIR / f"{manifest_name}.json"
    case = load_manifest(path)
    assert case.family == "cosmos3"
    # Re-read the raw JSON to verify the model_type field is matched by the
    # plugin (E2ECase doesn't surface model_type directly).
    import json
    with open(path) as f:
        raw = json.load(f)
    plugin = Cosmos3Plugin()
    assert plugin.matches(raw["model_type"])


@pytest.mark.parametrize("manifest_name", [
    "cosmos3-super",
    "cosmos3-super-reasoner",
    "cosmos3-nano",
])
def test_manifest_runtime_strategy(manifest_name):
    path = MANIFEST_DIR / f"{manifest_name}.json"
    case = load_manifest(path)
    assert case.runtime_strategy == "diffusion_cosmos3"
    assert case.runtime_strategy == Cosmos3Plugin.runtime_strategy


@pytest.mark.parametrize("manifest_name", [
    "cosmos3-super",
    "cosmos3-super-reasoner",
    "cosmos3-nano",
])
def test_manifest_has_skip_during_bringup(manifest_name):
    """While Phase 4 + 6 are incomplete, every cosmos3 manifest must skip."""
    path = MANIFEST_DIR / f"{manifest_name}.json"
    import json
    with open(path) as f:
        raw = json.load(f)
    assert raw.get("skip"), (
        f"{manifest_name} must skip until DM generator graph (Phase 4) "
        f"and C++ orchestration (Phase 6) land; got skip={raw.get('skip')!r}"
    )
