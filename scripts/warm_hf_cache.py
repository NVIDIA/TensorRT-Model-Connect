#!/usr/bin/env python3
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

try:
    from huggingface_hub import constants as hf_constants
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
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
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
_TTS_ASR_VERIFIER_MODEL = os.environ.get(
    "TRTMC_TTS_ASR_MODEL",
    "openai/whisper-large-v3-turbo",
)
_MAGPIE_REFERENCE_DEPENDENCIES = [
    (
        "magpie-nanocodec",
        "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
    ),
    (
        "magpie-byt5-tokenizer",
        "google/byt5-small",
    ),
    (
        "magpie-wavlm-discriminator",
        "microsoft/wavlm-base-plus",
    ),
]
_SANA_WM_HF_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
_SANA_WM_METADATA_ALLOW_PATTERNS = ["README.md", "config.yaml"]
_SANA_WM_FULL_ALLOW_PATTERNS = [
    "README.md",
    "model_index.json",
    "config.yaml",
    "pipeline*.py",
    "asset/sana_wm/**",
    "inference_video_scripts/**",
    "scheduler/**",
    "dit/**",
    "vae/**",
    "text_encoder/**",
    "refiner/**",
]

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

ROOT = pathlib.Path(__file__).resolve().parent.parent
manifests = sorted((ROOT / "tests" / "e2e" / "models").glob("*.json"))

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
needs_tts_asr_verifier = False
for m in manifests:
    d = json.loads(m.read_text())
    name = d.get("name", m.stem)
    if d.get("skip"):
        continue
    if filter_names is None and d.get("ci_tier") in excluded_ci_tiers:
        continue
    if not d.get("hf_id"):
        continue
    if filter_names is not None and name not in filter_names:
        continue
    entries.append((name, d["hf_id"], bool(d.get("gated"))))
    if str(d.get("runtime_strategy", "")).startswith("text_to_audio"):
        needs_tts_asr_verifier = True
        if str(d.get("family", "")) == "magpie_tts":
            # The NeMo Magpie reference restores the NanoCodec model, whose
            # discriminator loads WavLM; Magpie tokenizer setup also loads ByT5.
            entries.extend(
                (name, hf_id, False)
                for name, hf_id in _MAGPIE_REFERENCE_DEPENDENCIES
            )
    if str(d.get("runtime_strategy", "")) == "speech_to_speech":
        entries.append(("personaplex-mimi-codec", "kyutai/mimi", False))

if needs_tts_asr_verifier and _TTS_ASR_VERIFIER_MODEL not in {hf_id for _, hf_id, _ in entries}:
    entries.append(("tts-asr-verifier", _TTS_ASR_VERIFIER_MODEL, False))

deduped_entries: list[tuple[str, str, bool]] = []
seen_hf_ids: set[str] = set()
for name, hf_id, gated in entries:
    if hf_id in seen_hf_ids:
        continue
    seen_hf_ids.add(hf_id)
    deduped_entries.append((name, hf_id, gated))
entries = deduped_entries


def _is_cached(hf_id: str) -> bool:
    """Return True if the model has a usable local snapshot.

    A snapshot directory alone is not enough: partial cache entries can contain
    only config/tokenizer metadata. The offline build phase needs at least one
    HF entrypoint config and at least one local weight artifact.
    """
    cache_dir = pathlib.Path(hf_constants.HF_HUB_CACHE)
    # HF cache layout: models--{org}--{model}/snapshots/{sha}/
    repo_dir = cache_dir / ("models--" + hf_id.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False

    snapshot_paths = [
        path for path in snapshots_dir.iterdir()
        if path.is_dir()
    ]
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text().strip()
        main_snapshot = snapshots_dir / commit
        if main_snapshot.is_dir():
            snapshot_paths = [main_snapshot] + [
                path for path in snapshot_paths
                if path != main_snapshot
            ]

    return any(_snapshot_has_required_files(path) for path in snapshot_paths)


def _snapshot_has_required_files(snapshot_dir: pathlib.Path) -> bool:
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
    if (snapshot_dir / "model_index.json").is_file():
        return has_entrypoint and has_weights and not _diffusers_missing_weight_components(
            snapshot_dir
        )
    return has_entrypoint and has_weights


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


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _allow_patterns_for_hf_id(hf_id: str) -> list[str]:
    if hf_id.rstrip("/") == _SANA_WM_HF_ID:
        if _truthy_env("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS"):
            return list(_SANA_WM_FULL_ALLOW_PATTERNS)
        # SANA-WM is unusually large. Keep CI cache warming aligned with the
        # builder default: fetch the YAML contract unless full weights are
        # explicitly requested.
        return list(_SANA_WM_METADATA_ALLOW_PATTERNS)
    return _HF_ALLOW_PATTERNS + _HF_EXTRA_ALLOW_PATTERNS


selective = filter_names is not None
scope = f"selective ({len(entries)} models)" if selective else f"all {len(entries)} models"
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
            allow_patterns=_allow_patterns_for_hf_id(hf_id),
        )
        if not _snapshot_has_required_files(pathlib.Path(local_dir)):
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

print()
if selective and skipped:
    print(f"Skipped {len(skipped)} already-cached models (no network calls).")
if warned:
    print(
        f"Warning: {len(warned)}/{len(entries)} models could not be warmed: {warned}",
        file=sys.stderr,
    )
    print("Parallel E2E phase may re-issue model_info() for these models.")
else:
    downloaded = len(entries) - len(skipped)
    if downloaded == 0:
        print("All models already cached — zero network calls.")
    else:
        print(f"Downloaded {downloaded} model(s) successfully.")
