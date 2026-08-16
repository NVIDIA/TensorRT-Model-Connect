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

By default, partial failures remain warnings for existing nightly behavior.
Pass --strict to fail when a selected snapshot or file cannot be cached.
"""

import argparse
import fnmatch
import json
import os
import pathlib
import subprocess
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
    from huggingface_hub.file_download import repo_folder_name
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
    "processor_config.json",
    ".gitattributes",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.txt",
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
_ENTRYPOINT_PATTERNS = [
    "config.json",
    "model_index.json",
    "*.yml",
    "*.yaml",
    "*/config.json",
]
_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pth",
    "*.nemo",
    "model.npz",
    "elf_params.npz",
    "checkpoint_*/manifest.ocdbt",
]
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


def _family_hf_warm_file_specs(
    family: object,
) -> list[tuple[str, str, str, str]]:
    from tensorrt_model_connect.families import family_hf_warm_file_specs

    return family_hf_warm_file_specs(family)


_REQUIRED_FILES_BY_HF_ID = _load_family_hf_required_files_by_id()
_HF_FAMILY_ALLOW_PATTERNS = _load_family_hf_allow_patterns()
_HF_DOWNLOAD_PATTERNS = (
    _HF_ALLOW_PATTERNS + _HF_FAMILY_ALLOW_PATTERNS + _HF_EXTRA_ALLOW_PATTERNS
)
_TOKENIZER_DOWNLOAD_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "*.spm",
]
_DOWNLOAD_WORKER = ROOT / "scripts" / "hf_cache_download_worker.py"
_DOWNLOAD_ATTEMPTS = 2
_DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 600.0


def _is_hf_file_cached(
    hf_id: str,
    filename: str,
    revision: str = "",
) -> bool:
    try:
        revision_kwargs = {"revision": revision} if revision else {}
        hf_hub_download(
            hf_id,
            filename=filename,
            local_files_only=True,
            **revision_kwargs,
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
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit nonzero if any selected snapshot or file cannot be cached.",
)
parser.add_argument(
    "--local-only",
    action="store_true",
    help="Check cache readiness without making network requests or changing the cache.",
)
parser.add_argument(
    "--emit-cache-repos",
    metavar="JSON",
    help="After a successful cache check, write the unique selected Hugging Face "
         "repository IDs and their canonical cache folders to this JSON file. "
         "This is used to construct a positive per-model cache view.",
)
parser.add_argument(
    "--attempt-timeout-seconds",
    type=float,
    default=_DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    metavar="SECONDS",
    help="Maximum wall time for each online download attempt (default: 600).",
)
parser.add_argument(
    "--fail-fast",
    action="store_true",
    help="Stop after the first failed item. Requires --strict or "
         "--emit-cache-repos so the early stop cannot report success.",
)
args = parser.parse_args()
if args.attempt_timeout_seconds <= 0:
    parser.error("--attempt-timeout-seconds must be greater than zero")
if args.fail_fast and not (args.strict or args.emit_cache_repos):
    parser.error("--fail-fast requires --strict or --emit-cache-repos")

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
if args.strict and filter_names is not None:
    manifest_names = {
        str(json.loads(manifest.read_text()).get("name") or manifest.stem)
        for manifest in manifests
    }
    missing_names = sorted(filter_names - manifest_names)
    if missing_names:
        print(
            "ERROR: selected model manifest(s) not found: " + ", ".join(missing_names),
            file=sys.stderr,
        )
        sys.exit(1)

entries: list[tuple[str, str, str, bool, bool]] = []
file_assets: list[tuple[str, str, str, str]] = []
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
    entries.append(
        (
            name,
            d["hf_id"],
            str(d.get("hf_revision", "") or "").strip(),
            bool(d.get("gated")),
            True,
        )
    )
    entries.extend(
        (
            dependency_name,
            dependency_hf_id,
            "",
            False,
            not dependency_name.endswith("-tokenizer"),
        )
        for dependency_name, dependency_hf_id in _family_hf_warm_dependencies(
            d.get("family", "")
        )
    )
    file_assets.extend(_family_hf_warm_file_specs(d.get("family", "")))
deduped_entries: list[tuple[str, str, str, bool, bool]] = []
entry_indexes: dict[tuple[str, str], int] = {}
for name, hf_id, revision, gated, require_weights in entries:
    entry_key = (hf_id, revision)
    existing_index = entry_indexes.get(entry_key)
    if existing_index is None:
        entry_indexes[entry_key] = len(deduped_entries)
        deduped_entries.append((name, hf_id, revision, gated, require_weights))
        continue
    old_name, _, _, old_gated, old_require_weights = deduped_entries[existing_index]
    deduped_entries[existing_index] = (
        old_name,
        hf_id,
        revision,
        old_gated or gated,
        old_require_weights or require_weights,
    )
entries = deduped_entries
def _dedupe_file_assets(
    assets: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    deduped: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for asset in assets:
        asset_name, asset_hf_id, filename, revision = asset
        asset_key = (asset_hf_id, filename, revision)
        if asset_key in seen:
            continue
        seen.add(asset_key)
        deduped.append((asset_name, asset_hf_id, filename, revision))
    return deduped


file_assets = _dedupe_file_assets(file_assets)


def _is_cached(
    hf_id: str,
    revision: str = "",
    require_weights: bool = True,
) -> bool:
    """Return True if the model has a usable local snapshot.

    A snapshot directory alone is not enough: partial cache entries can contain
    only config/tokenizer metadata, and orphan snapshots without the requested
    revision ref cannot be resolved by the offline build phase.  Use
    ``snapshot_download(local_files_only=True)`` here so the warm-cache skip
    decision follows the same Hugging Face cache resolution path as the later
    offline builder.
    """
    try:
        revision_kwargs = {"revision": revision} if revision else {}
        local_dir = snapshot_download(
            hf_id,
            allow_patterns=(
                _HF_DOWNLOAD_PATTERNS
                if require_weights
                else _TOKENIZER_DOWNLOAD_PATTERNS
            ),
            local_files_only=True,
            **revision_kwargs,
        )
    except Exception:
        return False

    snapshot_dir = pathlib.Path(local_dir)
    if snapshot_dir.parent.name != "snapshots":
        return False
    return _snapshot_has_required_files(
        snapshot_dir,
        hf_id=hf_id,
        require_weights=require_weights,
    )


def _snapshot_has_required_files(
    snapshot_dir: pathlib.Path,
    hf_id: str = "",
    require_weights: bool = True,
) -> bool:
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
    if not require_weights:
        return has_entrypoint and has_required_files
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


def _worker_command(
    operation: str,
    hf_id: str,
    *,
    revision: str = "",
    allow_patterns: list[str] | None = None,
    filename: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(_DOWNLOAD_WORKER),
        "--operation",
        operation,
        "--repo-id",
        hf_id,
    ]
    if revision:
        command.extend(["--revision", revision])
    if operation == "snapshot":
        command.extend(["--allow-patterns-json", json.dumps(allow_patterns or [])])
    elif operation == "file":
        command.extend(["--filename", filename])
    else:
        raise ValueError(f"unsupported download worker operation: {operation}")
    return command


def _worker_failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    output = completed.stderr.strip() or completed.stdout.strip()
    if len(output) > 2000:
        output = "..." + output[-1997:]
    suffix = f": {output}" if output else ""
    return f"worker exited with status {completed.returncode}{suffix}"


def _run_download_worker(
    operation: str,
    hf_id: str,
    *,
    timeout_seconds: float,
    disable_xet: bool,
    revision: str = "",
    allow_patterns: list[str] | None = None,
    filename: str = "",
) -> tuple[str | None, str]:
    environment = os.environ.copy()
    if disable_xet:
        # This is applied to a fresh interpreter, before it imports the Hub.
        environment["HF_HUB_DISABLE_XET"] = "1"
    else:
        # Attempt one is always the default backend, even if the runner or
        # container image has a process-wide fallback override.
        environment.pop("HF_HUB_DISABLE_XET", None)
    command = _worker_command(
        operation,
        hf_id,
        revision=revision,
        allow_patterns=allow_patterns,
        filename=filename,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stderr or exc.stdout or "").strip()
        suffix = f"; last worker output: {output[-1000:]}" if output else ""
        return None, f"timed out and was killed after {timeout_seconds:g}s{suffix}"
    if completed.returncode != 0:
        return None, _worker_failure_detail(completed)
    try:
        payload = json.loads(completed.stdout)
        path = payload["path"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, f"worker returned invalid success evidence: {exc}"
    if not isinstance(path, str) or not path:
        return None, "worker returned an empty or non-string cache path"
    return path, ""


def _download_validation_error(
    operation: str,
    path: str,
    *,
    hf_id: str,
    require_weights: bool,
) -> str:
    downloaded_path = pathlib.Path(path)
    if operation == "file":
        return "" if downloaded_path.is_file() else f"downloaded file is missing: {path}"
    if not downloaded_path.is_dir():
        return f"downloaded snapshot directory is missing: {path}"
    if _snapshot_has_required_files(
        downloaded_path,
        hf_id=hf_id,
        require_weights=require_weights,
    ):
        return ""
    return (
        "downloaded snapshot is incomplete: missing a config/model_index "
        "entrypoint, required local weight artifact, or required component"
    )


def _run_download_attempts(
    operation: str,
    hf_id: str,
    *,
    timeout_seconds: float,
    revision: str = "",
    allow_patterns: list[str] | None = None,
    filename: str = "",
    require_weights: bool = True,
) -> tuple[str | None, str]:
    attempt_details: list[str] = []
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        disable_xet = attempt == _DOWNLOAD_ATTEMPTS
        started = time.monotonic()
        path, error = _run_download_worker(
            operation,
            hf_id,
            timeout_seconds=timeout_seconds,
            disable_xet=disable_xet,
            revision=revision,
            allow_patterns=allow_patterns,
            filename=filename,
        )
        elapsed = time.monotonic() - started
        mode = "Xet disabled" if disable_xet else "default transfer backend"
        prefix = (
            f"{hf_id}: attempt {attempt}/{_DOWNLOAD_ATTEMPTS} "
            f"({mode}, {elapsed:.1f}s)"
        )
        if error:
            attempt_details.append(f"{prefix} failed: {error}")
            continue
        try:
            validation_error = _download_validation_error(
                operation,
                path or "",
                hf_id=hf_id,
                require_weights=require_weights,
            )
        except Exception as exc:  # noqa: BLE001 - validation must fail closed
            validation_error = f"cache validation raised {type(exc).__name__}: {exc}"
        if validation_error:
            attempt_details.append(f"{prefix} failed validation: {validation_error}")
            continue
        attempt_details.append(f"{prefix} succeeded")
        return path, "; ".join(attempt_details)
    return None, "; ".join(attempt_details)


def _warm_snapshot(
    hf_id: str,
    *,
    revision: str = "",
    gated: bool,
    token_available: bool,
    selective: bool,
    local_only: bool = False,
    require_weights: bool = True,
    timeout_seconds: float = _DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    """Resolve locally first, downloading only when the cache is incomplete."""
    if (selective or local_only) and _is_cached(
        hf_id,
        revision=revision,
        require_weights=require_weights,
    ):
        return "cached", ""
    if local_only:
        return "failed", "required snapshot is not available in the local cache"
    if gated and not token_available:
        return "failed", "gated model is not cached and no HF token is available"
    local_dir, detail = _run_download_attempts(
        "snapshot",
        hf_id,
        timeout_seconds=timeout_seconds,
        revision=revision,
        allow_patterns=(
            _HF_DOWNLOAD_PATTERNS
            if require_weights
            else _TOKENIZER_DOWNLOAD_PATTERNS
        ),
        require_weights=require_weights,
    )
    return ("downloaded", detail) if local_dir else ("failed", detail)


def _warm_file(
    hf_id: str,
    filename: str,
    *,
    revision: str = "",
    selective: bool,
    local_only: bool,
    timeout_seconds: float = _DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    if (selective or local_only) and _is_hf_file_cached(
        hf_id,
        filename,
        revision=revision,
    ):
        return "cached", ""
    if local_only:
        return "failed", "required file is not available in the local cache"
    local_path, detail = _run_download_attempts(
        "file",
        hf_id,
        timeout_seconds=timeout_seconds,
        revision=revision,
        filename=filename,
    )
    return ("downloaded", detail) if local_path else ("failed", detail)


def _warm_exit_code(strict: bool, failures: list[str]) -> int:
    return 1 if strict and failures else 0


def _cache_repository_manifest(
    repo_ids: list[str],
    *,
    hub_cache: pathlib.Path,
) -> dict[str, object]:
    """Return a fail-closed manifest for selected repositories in one HF cache."""
    try:
        canonical_hub = hub_cache.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"HF Hub cache is unavailable: {hub_cache}: {exc}") from exc
    if not canonical_hub.is_dir():
        raise RuntimeError(f"HF Hub cache is not a directory: {canonical_hub}")

    repositories: list[dict[str, str]] = []
    seen_repo_ids: set[str] = set()
    seen_folders: set[str] = set()
    for repo_id in repo_ids:
        if repo_id in seen_repo_ids:
            continue
        seen_repo_ids.add(repo_id)
        folder = repo_folder_name(repo_id=repo_id, repo_type="model")
        if not folder or "/" in folder or "\\" in folder or folder in {".", ".."}:
            raise RuntimeError(
                f"Hugging Face repository has an unsafe canonical cache folder: "
                f"{repo_id!r}: {folder!r}"
            )
        if folder in seen_folders:
            raise RuntimeError(f"duplicate canonical cache folder: {folder}")
        seen_folders.add(folder)

        raw_repo_path = canonical_hub / folder
        if raw_repo_path.is_symlink() or not raw_repo_path.is_dir():
            raise RuntimeError(
                f"selected Hugging Face repository cache is missing or not a directory: "
                f"{repo_id}: {raw_repo_path}"
            )
        try:
            canonical_repo_path = raw_repo_path.resolve(strict=True)
            canonical_repo_path.relative_to(canonical_hub)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"selected Hugging Face repository cache escapes the configured hub: "
                f"{repo_id}: {raw_repo_path}"
            ) from exc
        repositories.append(
            {
                "repo_id": repo_id,
                "repo_type": "model",
                "cache_folder": folder,
                "cache_path": str(canonical_repo_path),
            }
        )

    if not repositories:
        raise RuntimeError("no selected Hugging Face repositories were resolved")
    return {
        "schema_version": 1,
        "hub_cache": str(canonical_hub),
        "repositories": repositories,
    }


def _write_cache_repository_manifest(
    output: pathlib.Path,
    repo_ids: list[str],
) -> None:
    payload = _cache_repository_manifest(
        repo_ids,
        hub_cache=pathlib.Path(hf_constants.HF_HUB_CACHE),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


selective = filter_names is not None
asset_scope = f", {len(file_assets)} file asset(s)" if file_assets else ""
scope = (
    f"selective ({len(entries)} models{asset_scope})"
    if selective
    else f"all {len(entries)} models{asset_scope}"
)
action = "Checking HF cache readiness" if args.local_only else "Warming HF cache"
print(f"{action} — {scope}...")
print(f"HF Hub cache: {hf_constants.HF_HUB_CACHE}")

warned: list[str] = []
skipped: list[str] = []
stopped_early = False
fail_closed = args.strict or bool(args.emit_cache_repos)
hf_token_available = bool(
    os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

for i, (name, hf_id, revision, gated, require_weights) in enumerate(entries, 1):
    status, detail = _warm_snapshot(
        hf_id,
        revision=revision,
        gated=gated,
        token_available=hf_token_available,
        selective=selective,
        local_only=args.local_only,
        require_weights=require_weights,
        timeout_seconds=args.attempt_timeout_seconds,
    )
    if status == "cached":
        print(f"  [{i:3d}/{len(entries)}] {name}  CACHED (skip)")
        skipped.append(name)
        continue
    if status == "downloaded":
        print(f"  [{i:3d}/{len(entries)}] {name}  OK: {detail}")
        # Small inter-request delay to stay well below the API rate limit.
        time.sleep(0.3)
        continue
    if gated and not hf_token_available:
        print(f"  [{i:3d}/{len(entries)}] {name}  SKIP ({detail})")
        if not fail_closed:
            skipped.append(name)
            continue
    else:
        print(f"  [{i:3d}/{len(entries)}] {name}  WARN: {detail}")
    if status == "failed":
        warned.append(name)
        if args.fail_fast:
            print(
                f"Fail-fast: stopping after exhausted item {name} ({hf_id}).",
                file=sys.stderr,
            )
            stopped_early = True
            break

if file_assets and not stopped_early:
    print()
    print("Warming family file assets...")
for i, (name, hf_id, filename, revision) in enumerate(
    file_assets if not stopped_early else [],
    1,
):
    status, detail = _warm_file(
        hf_id,
        filename,
        revision=revision,
        selective=selective,
        local_only=args.local_only,
        timeout_seconds=args.attempt_timeout_seconds,
    )
    if status == "cached":
        print(f"  [{i:3d}/{len(file_assets)}] {name}  CACHED (skip)")
        skipped.append(name)
        continue
    if status == "downloaded":
        print(f"  [{i:3d}/{len(file_assets)}] {name}  OK: {detail}")
        time.sleep(0.3)
        continue
    print(f"  [{i:3d}/{len(file_assets)}] {name}  WARN: {detail}")
    warned.append(name)
    if args.fail_fast:
        print(
            f"Fail-fast: stopping after exhausted item {name} ({hf_id}/{filename}).",
            file=sys.stderr,
        )
        stopped_early = True
        break

print()
if selective and skipped:
    print(f"Skipped {len(skipped)} already-cached item(s) (no network calls).")
if warned:
    total_items = len(entries) + len(file_assets)
    failure_action = "are missing from the cache" if args.local_only else "could not be warmed"
    print(
        f"Warning: {len(warned)}/{total_items} item(s) {failure_action}: {warned}",
        file=sys.stderr,
    )
    if not args.local_only:
        print("Parallel E2E phase may re-issue HF cache requests for these item(s).")
else:
    downloaded = len(entries) + len(file_assets) - len(skipped)
    if downloaded == 0:
        print("All items already cached — zero network calls.")
    else:
        print(f"Downloaded {downloaded} item(s) successfully.")

if args.emit_cache_repos and not warned:
    selected_repo_ids = [hf_id for _, hf_id, _, _, _ in entries]
    selected_repo_ids.extend(hf_id for _, hf_id, _, _ in file_assets)
    try:
        _write_cache_repository_manifest(
            pathlib.Path(args.emit_cache_repos),
            selected_repo_ids,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not emit selected cache repositories: {exc}", file=sys.stderr)
        warned.append("cache-repository-evidence")
    else:
        print(f"Selected cache repository evidence: {args.emit_cache_repos}")

strict_exit_code = _warm_exit_code(fail_closed, warned)
if strict_exit_code:
    sys.exit(strict_exit_code)
