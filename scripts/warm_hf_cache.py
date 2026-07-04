#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm the HF Hub file/metadata cache before a parallel E2E rebuild phase.

Two operating modes, selected by --models-file:

Nightly mode (no --models-file):
    Calls snapshot_download() for every non-skipped E2E model.  This refreshes
    the cached commit SHA (model_info() call) and downloads any stale/missing
    files, so the parallel rebuild phase can set HF_HUB_OFFLINE=1 safely.
    Sequential with a 0.3 s inter-request delay to stay below the HF API rate
    limit (10 k requests / 5 min).

PR CI selective mode (--models-file FILE):
    For each model in FILE:
      - Already in cache → skip entirely (zero network calls).
      - Not in cache     → call snapshot_download() to download it.
    This handles newly added model families whose weights are not yet in the
    persistent cache without making unnecessary API calls for the majority of
    models that are already cached.

Usage:
    # Nightly — warm all non-skipped E2E models:
    python scripts/warm_hf_cache.py

    # PR CI — only download models missing from cache:
    python scripts/warm_hf_cache.py --models-file e2e_models.txt

Exit code 0 even on partial failures — missing cache entries produce a warning
but do not block CI.
"""

import argparse
import fnmatch
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_PYTHON = ROOT / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

try:
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not available", file=sys.stderr)
    sys.exit(1)

# Keep this intentionally aligned with tensorrt_model_connect.engine_builder._HF_ALLOW_PATTERNS
# without importing engine_builder here; that import pulls in the whole builder
# plugin registry before the cache warm script needs it.
_HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "linear_spec_lora/**",
    "*.model",
    "*.spm",
    "*.py",
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "transformer/**",
    "vae/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "*/config.json",
    "*/model.safetensors",
    "*/model-*.safetensors",
    "*/model.safetensors.index.json",
    "*/diffusion_pytorch_model.safetensors",
    "*/diffusion_pytorch_model-*.safetensors",
    "*/diffusion_pytorch_model.safetensors.index.json",
    "scheduler/*",
    "tokenizer/*",
]

_HF_EXTRA_ALLOW_PATTERNS = ["*.nemo"]
_ENTRYPOINT_PATTERNS = ["config.json", "model_index.json", "*/config.json"]
_WEIGHT_PATTERNS = ["*.safetensors", "*.bin", "*.nemo"]
_DIFFUSERS_WEIGHT_COMPONENTS = {
    "controlnet",
    "image_encoder",
    "text_encoder",
    "text_encoder_2",
    "transformer",
    "unet",
    "vae",
}


def _load_family_hf_required_files_by_id() -> dict[str, list[str]]:
    from tensorrt_model_connect.families import family_hf_required_files_by_id

    return family_hf_required_files_by_id()


def _load_family_hf_allow_patterns() -> list[str]:
    from tensorrt_model_connect.families import family_hf_allow_patterns

    return family_hf_allow_patterns()


def _family_hf_warm_dependencies(family: object) -> list[tuple[str, str]]:
    from tensorrt_model_connect.families import family_hf_warm_dependencies

    return family_hf_warm_dependencies(family)


def _family_hf_warm_files(family: object) -> list[tuple[str, str, str]]:
    from tensorrt_model_connect.families import family_hf_warm_files

    return family_hf_warm_files(family)


_REQUIRED_FILES_BY_HF_ID = _load_family_hf_required_files_by_id()
_HF_FAMILY_ALLOW_PATTERNS = _load_family_hf_allow_patterns()
_HF_DOWNLOAD_PATTERNS = (
    _HF_ALLOW_PATTERNS + _HF_FAMILY_ALLOW_PATTERNS + _HF_EXTRA_ALLOW_PATTERNS
)


def _is_hf_file_cached(hf_id: str, filename: str) -> bool:
    try:
        hf_hub_download(
            hf_id,
            filename=filename,
            local_files_only=True,
        )
    except Exception:
        return False
    return True


def _manifest_has_eligible_testcase(
    manifest: dict, excluded_ci_tiers: set[str]
) -> bool:
    testcases = manifest.get("testcases")
    if not isinstance(testcases, list) or not testcases:
        return str(manifest.get("ci_tier") or "") not in excluded_ci_tiers
    return any(
        isinstance(testcase, dict)
        and str(testcase.get("ci_tier") or "") not in excluded_ci_tiers
        for testcase in testcases
    )


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--models-file",
    metavar="FILE",
    help="Path to a file with one model name per line (manifest stems). "
         "When given, only those models are considered and already-cached "
         "models are skipped (no network call). Intended for PR CI selective "
         "warm.",
)
parser.add_argument(
    "--exclude-ci-tier",
    action="append",
    default=[],
    help="Exclude manifests with this ci_tier value. Intended for nightly mode "
         "to skip PR-only representative manifests.",
)
args = parser.parse_args()

models_dir = ROOT / "tests" / "e2e" / "models"
manifests = sorted({
    *models_dir.glob("*.json"),
    *models_dir.glob("*/manifests/*.json"),
})

# Optional filter: only consider models listed in --models-file
filter_names: set[str] | None = None
if args.models_file:
    p = pathlib.Path(args.models_file)
    if not p.is_file():
        print(f"ERROR: --models-file {p} not found", file=sys.stderr)
        sys.exit(1)
    filter_names = {line.strip() for line in p.read_text().splitlines() if line.strip()}
excluded_ci_tiers = set(args.exclude_ci_tier or [])

entries: list[tuple[str, str, bool]] = []
file_assets: list[tuple[str, str, str]] = []
for m in manifests:
    d = json.loads(m.read_text())
    name = d.get("name", m.stem)
    if d.get("skip"):
        continue
    if not _manifest_has_eligible_testcase(d, excluded_ci_tiers):
        continue
    if not d.get("hf_id"):
        continue
    if filter_names is not None and name not in filter_names:
        continue
    entries.append((name, d["hf_id"], bool(d.get("gated"))))
    entries.extend(
        (dependency_name, dependency_hf_id, False)
        for dependency_name, dependency_hf_id in _family_hf_warm_dependencies(
            d.get("family", "")
        )
    )
    file_assets.extend(_family_hf_warm_files(d.get("family", "")))
deduped_entries: list[tuple[str, str, bool]] = []
seen_hf_ids: set[str] = set()
for name, hf_id, gated in entries:
    if hf_id in seen_hf_ids:
        continue
    seen_hf_ids.add(hf_id)
    deduped_entries.append((name, hf_id, gated))
entries = deduped_entries
deduped_file_assets: list[tuple[str, str, str]] = []
seen_file_assets: set[tuple[str, str]] = set()
for asset_name, asset_hf_id, filename in file_assets:
    asset_key = (asset_hf_id, filename)
    if asset_key in seen_file_assets:
        continue
    seen_file_assets.add(asset_key)
    deduped_file_assets.append((asset_name, asset_hf_id, filename))
file_assets = deduped_file_assets


def _is_cached(hf_id: str) -> bool:
    """Return True if the model has a usable local snapshot.

    A snapshot directory alone is not enough: partial cache entries can contain
    only config/tokenizer metadata, and orphan snapshots without the requested
    revision ref cannot be resolved by the offline build phase.  Use
    ``snapshot_download(local_files_only=True)`` here so the warm-cache skip
    decision follows the same Hugging Face cache resolution path as the later
    offline builder.
    """
    try:
        local_dir = snapshot_download(
            hf_id,
            allow_patterns=_HF_DOWNLOAD_PATTERNS,
            local_files_only=True,
        )
    except Exception:
        return False

    snapshot_dir = pathlib.Path(local_dir)
    if snapshot_dir.parent.name != "snapshots":
        return False
    return _snapshot_has_required_files(snapshot_dir, hf_id=hf_id)


def _snapshot_has_required_files(snapshot_dir: pathlib.Path, hf_id: str = "") -> bool:
    files = [
        str(path.relative_to(snapshot_dir))
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    ]
    if any(fnmatch.fnmatch(name, "*.nemo") for name in files):
        return True
    has_entrypoint = any(
        fnmatch.fnmatch(name, pattern)
        for name in files
        for pattern in _ENTRYPOINT_PATTERNS
    )
    has_weights = any(
        fnmatch.fnmatch(name, pattern)
        for name in files
        for pattern in _WEIGHT_PATTERNS
    )
    required_files = _REQUIRED_FILES_BY_HF_ID.get(hf_id, [])
    has_required_files = all((snapshot_dir / name).is_file() for name in required_files)
    if (snapshot_dir / "model_index.json").is_file():
        return has_entrypoint and has_weights and has_required_files and not _diffusers_missing_weight_components(
            snapshot_dir
        )
    return has_entrypoint and has_weights and has_required_files


def _diffusers_missing_weight_components(snapshot_dir: pathlib.Path) -> list[str]:
    model_index_path = snapshot_dir / "model_index.json"
    try:
        model_index = json.loads(model_index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ["model_index.json"]

    required_components = sorted(
        name for name, value in model_index.items()
        if (
            name in _DIFFUSERS_WEIGHT_COMPONENTS
            and _is_diffusers_component_enabled(value)
        )
    )
    return [
        component for component in required_components
        if not _component_has_weight(snapshot_dir, component)
    ]


def _is_diffusers_component_enabled(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, list) and all(item is None for item in value):
        return False
    return True


def _component_has_weight(snapshot_dir: pathlib.Path, component: str) -> bool:
    component_dir = snapshot_dir / component
    if not component_dir.is_dir():
        return False
    return any(
        path.is_file()
        and any(fnmatch.fnmatch(path.name, pattern) for pattern in _WEIGHT_PATTERNS)
        for path in component_dir.rglob("*")
    )


selective = filter_names is not None
asset_scope = f", {len(file_assets)} file asset(s)" if file_assets else ""
scope = (
    f"selective ({len(entries)} models{asset_scope})"
    if selective
    else f"all {len(entries)} models{asset_scope}"
)
print(f"Warming HF cache — {scope}...")
print(f"HF Hub cache: {hf_constants.HF_HUB_CACHE}")

warned: list[str] = []
skipped: list[str] = []
hf_token_available = bool(
    os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

for i, (name, hf_id, gated) in enumerate(entries, 1):
    if gated and not hf_token_available:
        print(f"  [{i:3d}/{len(entries)}] {name}  SKIP (gated, no HF token)")
        skipped.append(name)
        continue
    if selective and _is_cached(hf_id):
        print(f"  [{i:3d}/{len(entries)}] {name}  CACHED (skip)")
        skipped.append(name)
        continue

    try:
        local_dir = snapshot_download(
            hf_id,
            allow_patterns=_HF_DOWNLOAD_PATTERNS,
        )
        if not _snapshot_has_required_files(pathlib.Path(local_dir), hf_id=hf_id):
            raise RuntimeError(
                "downloaded snapshot is still missing a config/model_index "
                "entrypoint or required local weight artifact")
        print(f"  [{i:3d}/{len(entries)}] {name}  OK")
    except HfHubHTTPError as e:
        print(f"  [{i:3d}/{len(entries)}] {name}  WARN (HTTP {e.response.status_code}): {e}")
        warned.append(name)
    except Exception as e:  # noqa: BLE001
        print(f"  [{i:3d}/{len(entries)}] {name}  WARN: {e}")
        warned.append(name)
    # Small inter-request delay to stay well below the API rate limit.
    time.sleep(0.3)

if file_assets:
    print()
    print("Warming family file assets...")
for i, (name, hf_id, filename) in enumerate(file_assets, 1):
    if selective and _is_hf_file_cached(hf_id, filename):
        print(f"  [{i:3d}/{len(file_assets)}] {name}  CACHED (skip)")
        skipped.append(name)
        continue

    try:
        hf_hub_download(hf_id, filename=filename)
        print(f"  [{i:3d}/{len(file_assets)}] {name}  OK")
    except HfHubHTTPError as e:
        print(f"  [{i:3d}/{len(file_assets)}] {name}  WARN (HTTP {e.response.status_code}): {e}")
        warned.append(name)
    except Exception as e:  # noqa: BLE001
        print(f"  [{i:3d}/{len(file_assets)}] {name}  WARN: {e}")
        warned.append(name)
    time.sleep(0.3)

print()
if selective and skipped:
    print(f"Skipped {len(skipped)} already-cached item(s) (no network calls).")
if warned:
    total_items = len(entries) + len(file_assets)
    print(
        f"Warning: {len(warned)}/{total_items} item(s) could not be warmed: {warned}",
        file=sys.stderr,
    )
    print("Parallel E2E phase may re-issue HF cache requests for these item(s).")
else:
    downloaded = len(entries) + len(file_assets) - len(skipped)
    if downloaded == 0:
        print("All items already cached — zero network calls.")
    else:
        print(f"Downloaded {downloaded} item(s) successfully.")
