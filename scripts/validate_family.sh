#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate a model family end-to-end: build bundle, diff logits, diff layers, runner parity.
#
# Usage:
#   ./scripts/validate_family.sh org/example-decoder                  # HF repo ID
#   ./scripts/validate_family.sh models/hf/org__example-decoder       # local dir
#   ./scripts/validate_family.sh org/example-decoder --max-cache-length 512
#   ./scripts/validate_family.sh org/example-decoder --binary ./build/trtmc
#   ./scripts/validate_family.sh org/example-decoder --engine-dir /tmp/trtmc-engines
#   ./scripts/validate_family.sh ./local-model --e2e-model example-decoder
#   ./scripts/validate_family.sh org/example-decoder --isolate-model-plugin
#
# Requirements: torch, tensorrt_model_connect installed, C++ binary built.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
MAX_CACHE_LENGTH=256
BINARY="${PROJECT_DIR}/build/trtmc"
BUNDLE_DIR="/tmp"
ENGINE_DIR="${ENGINE_DIR:-}"
TRUST_REMOTE_CODE_ARGS=()
MODEL_PLUGIN_DIR=""
ISOLATE_MODEL_PLUGIN="false"
E2E_MODEL_SELECTOR=""

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cache-length) MAX_CACHE_LENGTH="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --engine-dir) ENGINE_DIR="$2"; shift 2 ;;
        --e2e-model) E2E_MODEL_SELECTOR="$2"; shift 2 ;;
        --model-plugin-dir) MODEL_PLUGIN_DIR="$2"; shift 2 ;;
        --isolate-model-plugin) ISOLATE_MODEL_PLUGIN="true"; shift ;;
        --trust-remote-code) TRUST_REMOTE_CODE_ARGS+=(--trust-remote-code); shift ;;
        -h|--help)
            echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--engine-dir DIR] [--e2e-model MANIFEST_NAME] [--model-plugin-dir DIR] [--isolate-model-plugin] [--trust-remote-code]"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            else
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: model ID or path required." >&2
    echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--engine-dir DIR] [--e2e-model MANIFEST_NAME] [--model-plugin-dir DIR] [--isolate-model-plugin] [--trust-remote-code]" >&2
    exit 1
fi

copy_isolated_model_plugin() {
    local runtime_strategy="$1"
    local binary_path="$2"
    local out_dir="$3"

    "$HF_PYTHON" - "$PROJECT_DIR" "$runtime_strategy" "$binary_path" "$out_dir" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

project = Path(sys.argv[1])
strategy = sys.argv[2]
binary = Path(sys.argv[3])
out_dir = Path(sys.argv[4])

matches: list[tuple[str, str]] = []
for index_path in sorted((project / "src" / "runtime" / "models").glob("*/MODEL.toml")):
    raw = tomllib.loads(index_path.read_text(encoding="utf-8"))
    strategies = raw.get("runtime_strategies") or []
    if strategy in strategies:
        model_id = str(raw.get("id") or index_path.parent.name)
        library = str(raw.get("runtime_library") or f"libtrtmc_model_{model_id}.so")
        matches.append((model_id, library))

if not matches:
    raise SystemExit(f"No runtime model plugin owns runtime_strategy={strategy!r}")
if len(matches) > 1:
    owners = ", ".join(model for model, _ in matches)
    raise SystemExit(f"Multiple runtime model plugins own runtime_strategy={strategy!r}: {owners}")

model_id, library = matches[0]
src = binary.parent / "models" / model_id / library
if not src.is_file():
    raise SystemExit(f"Runtime model plugin library not found: {src}")

out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / library
shutil.copy2(src, dst)
print(out_dir)
PY
}

runtime_strategy_has_validation_profile() {
    local runtime_strategy="$1"
    local profile="$2"

    "$HF_PYTHON" - "$PROJECT_DIR" "$runtime_strategy" "$profile" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

project = Path(sys.argv[1])
strategy = sys.argv[2]
profile = sys.argv[3]

for manifest_path in sorted((project / "src" / "runtime" / "models").glob("*/MODEL.toml")):
    raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = raw.get("validation_profiles") or {}
    strategies = profiles.get(profile) if isinstance(profiles, dict) else None
    if isinstance(strategies, list) and strategy in strategies:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

resolve_e2e_manifest() {
    "$HF_PYTHON" - "$PROJECT_DIR" "$MODEL" "$E2E_MODEL_SELECTOR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

project = Path(sys.argv[1])
model = sys.argv[2]
selector = sys.argv[3]
models_dir = project / "tests" / "e2e" / "models"
matches: list[tuple[str, str, str]] = []

for index_path in sorted(models_dir.glob("*/MODEL.toml")):
    index = tomllib.loads(index_path.read_text(encoding="utf-8"))
    family = str(index.get("id") or index_path.parent.name)
    entries = index.get("test_manifests") or []
    if not isinstance(entries, list):
        raise SystemExit(f"{index_path}: test_manifests must be a list")
    for entry in entries:
        if not isinstance(entry, str):
            raise SystemExit(f"{index_path}: test_manifests entries must be strings")
        relative = PurePosixPath(entry.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"{index_path}: invalid manifest path {entry!r}")
        manifest_path = index_path.parent / Path(*relative.parts)
        if not manifest_path.is_file():
            raise SystemExit(f"{index_path}: missing indexed manifest {entry!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("skip"):
            continue
        name = str(manifest.get("name") or "")
        hf_id = str(manifest.get("hf_id") or manifest.get("model_id") or "")
        bundle = str(manifest.get("bundle") or (f"{name}.trtfb" if name else ""))
        if not name or not bundle:
            raise SystemExit(f"{manifest_path}: name and bundle must be non-empty")
        bundle_path = PurePosixPath(bundle.replace("\\", "/"))
        if bundle_path.is_absolute() or len(bundle_path.parts) != 1 or ".." in bundle_path.parts:
            raise SystemExit(f"{manifest_path}: bundle must be a plain filename")
        if (selector and name == selector) or (not selector and hf_id == model):
            matches.append((name, family, bundle))

if not matches:
    if selector:
        raise SystemExit(f"No active indexed E2E manifest has name={selector!r}")
    raise SystemExit(
        f"No active indexed E2E manifest has exact hf_id={model!r}; "
        "local checkpoints require --e2e-model MANIFEST_NAME"
    )
if len(matches) > 1:
    names = ", ".join(sorted(name for name, _family, _bundle in matches))
    raise SystemExit(
        f"E2E manifest selection is ambiguous for {selector or model!r}: {names}; "
        "select one with --e2e-model MANIFEST_NAME"
    )

name, family, bundle = matches[0]
for value in (name, family, bundle):
    if any(character in value for character in "\t\r\n"):
        raise SystemExit("E2E manifest fields must not contain tabs or newlines")
print(f"{name}\t{family}\t{bundle}")
PY
}

# Derive a safe bundle filename from the model ID.
SAFE_NAME="$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')"
BUNDLE_PATH="${BUNDLE_DIR}/${SAFE_NAME}.trtfb"
ENGINE_DIR="${ENGINE_DIR:-${BUNDLE_DIR}/engines}"

# Container-baked Python with HF/TRT deps
HF_PYTHON="${HF_PYTHON:-/opt/venv/bin/python}"
if [[ ! -x "$HF_PYTHON" ]]; then
    echo "ERROR: HF python not found at $HF_PYTHON" >&2
    echo "Run inside the dev container, or override HF_PYTHON=/path/to/python." >&2
    exit 1
fi

# Set up LD_LIBRARY_PATH for TRT
TRT_LIB_DIR=$("$HF_PYTHON" -c "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])" 2>/dev/null || true)
if [[ -n "$TRT_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${TRT_LIB_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

PASS=0
FAIL=0
STEPS=()
BUILD_STAGE_DIR=""
CANDIDATE_BUNDLE_PATH=""
BUNDLE_INSPECTION=""

run_step() {
    local name="$1"
    shift
    echo ""
    echo "==== $name ===="
    if "$@"; then
        STEPS+=("PASS  $name")
        PASS=$((PASS + 1))
    else
        STEPS+=("FAIL  $name")
        FAIL=$((FAIL + 1))
    fi
}

cleanup_build_stage() {
    if [[ -z "$BUILD_STAGE_DIR" ]] || [[ ! -d "$BUILD_STAGE_DIR" ]]; then
        return
    fi
    rm -f -- \
        "$CANDIDATE_BUNDLE_PATH" \
        "${CANDIDATE_BUNDLE_PATH%.trtfb}.effective_config.json"
    rmdir -- "$BUILD_STAGE_DIR" 2>/dev/null || true
}

# Step 1: Build bundle
mkdir -p "$BUNDLE_DIR"
BUILD_STAGE_DIR="$(mktemp -d "${BUNDLE_DIR%/}/.validate-family-build.XXXXXX")"
CANDIDATE_BUNDLE_PATH="${BUILD_STAGE_DIR}/$(basename "$BUNDLE_PATH")"
trap cleanup_build_stage EXIT
BUILD_ARGS=(
    build "$MODEL"
    -o "$CANDIDATE_BUNDLE_PATH"
    --max-cache-length "$MAX_CACHE_LENGTH"
    "${TRUST_REMOTE_CODE_ARGS[@]}"
)

build_candidate_bundle() {
    if ! "$BINARY" "${BUILD_ARGS[@]}"; then
        return 1
    fi
    if [[ ! -s "$CANDIDATE_BUNDLE_PATH" ]]; then
        echo "ERROR: build returned success without writing a non-empty bundle: $CANDIDATE_BUNDLE_PATH" >&2
        return 1
    fi
    if ! BUNDLE_INSPECTION="$("$BINARY" inspect "$CANDIDATE_BUNDLE_PATH")"; then
        echo "ERROR: the bundle produced by this invocation cannot be inspected: $CANDIDATE_BUNDLE_PATH" >&2
        return 1
    fi
    return 0
}

BUILD_FAILURES_BEFORE="$FAIL"
run_step "Build bundle" build_candidate_bundle
BUILD_SUCCEEDED="false"
if [[ "$FAIL" -eq "$BUILD_FAILURES_BEFORE" ]]; then
    BUILD_SUCCEEDED="true"
fi

# Detect runtime strategy from the built bundle to skip decoder-only tools
# for encoder-only / seq2seq models (diff_logits, diff_layers, parity only
# work with decoder models that use TrtRunner).
if [[ "$BUILD_SUCCEEDED" == "true" ]]; then
    RUNTIME_STRATEGY=$(printf '%s\n' "$BUNDLE_INSPECTION" \
        | grep -i "runtime strategy" | awk '{print $NF}' || true)
else
    RUNTIME_STRATEGY=""
fi

if [[ "$ISOLATE_MODEL_PLUGIN" == "true" ]]; then
    if [[ -z "$RUNTIME_STRATEGY" ]]; then
        echo ""
        echo "==== isolate model plugin ===="
        echo "FAIL: cannot isolate plugin because runtime strategy could not be read"
        STEPS+=("FAIL  isolate model plugin (missing runtime strategy)")
        FAIL=$((FAIL + 1))
    elif [[ ! -x "$BINARY" ]]; then
        echo ""
        echo "==== isolate model plugin ===="
        echo "FAIL: cannot isolate plugin because C++ binary is not executable: $BINARY"
        STEPS+=("FAIL  isolate model plugin (no binary)")
        FAIL=$((FAIL + 1))
    else
        ONLY_DIR="${BUNDLE_DIR}/only-${SAFE_NAME}"
        echo ""
        echo "==== isolate model plugin ===="
        if MODEL_PLUGIN_DIR="$(copy_isolated_model_plugin "$RUNTIME_STRATEGY" "$BINARY" "$ONLY_DIR")"; then
            echo "Using isolated model plugin dir: $MODEL_PLUGIN_DIR"
            export TRTMC_MODEL_PLUGIN_DIR="$MODEL_PLUGIN_DIR"
            export TRTMC_MODEL_PLUGIN_STRICT=1
            STEPS+=("PASS  isolate model plugin (${RUNTIME_STRATEGY})")
            PASS=$((PASS + 1))
        else
            echo "FAIL: unable to prepare isolated model plugin dir"
            STEPS+=("FAIL  isolate model plugin (${RUNTIME_STRATEGY})")
            FAIL=$((FAIL + 1))
        fi
    fi
fi
IS_DECODER=false
if runtime_strategy_has_validation_profile "$RUNTIME_STRATEGY" "decoder_debug"; then
    IS_DECODER=true
fi

if [[ "$IS_DECODER" == "true" ]]; then
    # Steps 2-4: decoder-only validation (diff_logits, diff_layers, runner parity)
    run_step "diff_logits (battery)" \
        "$HF_PYTHON" "${PROJECT_DIR}/tools/diff_logits.py" \
            --model "$MODEL" --atol 1e-3 --battery \
            --max-cache-length "$MAX_CACHE_LENGTH" "${TRUST_REMOTE_CODE_ARGS[@]}"

    run_step "diff_layers" \
        "$HF_PYTHON" "${PROJECT_DIR}/tools/diff_layers.py" \
            --model "$MODEL" --atol 0.05 \
            --max-cache-length "$MAX_CACHE_LENGTH" "${TRUST_REMOTE_CODE_ARGS[@]}"

    if [[ -x "$BINARY" ]]; then
        run_step "test_runner_parity" \
            "$HF_PYTHON" "${PROJECT_DIR}/tools/test_runner_parity.py" \
                --bundle "$CANDIDATE_BUNDLE_PATH" --binary "$BINARY" \
                --hf-python "$HF_PYTHON" --max-new-tokens 20
    else
        echo ""
        echo "==== test_runner_parity ===="
        echo "SKIP: C++ binary not found at $BINARY"
        STEPS+=("SKIP  test_runner_parity (no binary)")
    fi
else
    echo ""
    echo "==== diff_logits / diff_layers / runner_parity ===="
    echo "SKIP: runtime_strategy='${RUNTIME_STRATEGY}' is not decoder-only"
    echo "      (diff tools only support decoder models; E2E pytest is the gate)"
    STEPS+=("SKIP  diff_logits (non-decoder: ${RUNTIME_STRATEGY})")
    STEPS+=("SKIP  diff_layers (non-decoder: ${RUNTIME_STRATEGY})")
    STEPS+=("SKIP  test_runner_parity (non-decoder: ${RUNTIME_STRATEGY})")
fi

# Step 5: E2E pytest. The harness must consume the bundle built above; it must
# never rebuild the manifest's canonical hf_id as a substitute for this model.
E2E_MODEL=""
E2E_FAMILY=""
E2E_BUNDLE=""
E2E_METADATA=""
if E2E_METADATA="$(resolve_e2e_manifest)"; then
    IFS=$'\t' read -r E2E_MODEL E2E_FAMILY E2E_BUNDLE <<<"$E2E_METADATA"
fi

if [[ -n "$E2E_MODEL" ]] && [[ -x "$BINARY" ]] && [[ "$BUILD_SUCCEEDED" == "true" ]]; then
    mkdir -p "$ENGINE_DIR"
    E2E_PROOF_DIR="$(mktemp -d "${ENGINE_DIR%/}/validate-current.XXXXXX")"
    BUNDLE_SOURCE="$(cd "$(dirname "$CANDIDATE_BUNDLE_PATH")" && pwd)/$(basename "$CANDIDATE_BUNDLE_PATH")"
    ln -s "$BUNDLE_SOURCE" "${E2E_PROOF_DIR}/${E2E_BUNDLE}"
    E2E_NODE="${PROJECT_DIR}/tests/e2e/models/${E2E_FAMILY}/test_${E2E_FAMILY}_e2e.py::test_model_e2e[${E2E_MODEL}]"
    E2E_ARGS=(
        "$E2E_NODE"
        -v
        --engine-dir "$E2E_PROOF_DIR"
        --trtmc-binary "$BINARY"
        --hf-python "$HF_PYTHON"
    )
    if [[ -n "$MODEL_PLUGIN_DIR" ]]; then
        E2E_ARGS+=(--model-plugin-dir "$MODEL_PLUGIN_DIR")
    fi
    run_step "E2E pytest [${E2E_MODEL}]" \
        "$HF_PYTHON" -m pytest "${E2E_ARGS[@]}"
    rm -f "${E2E_PROOF_DIR}/${E2E_BUNDLE}"
    rmdir "$E2E_PROOF_DIR" 2>/dev/null || true
elif [[ -z "$E2E_MODEL" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "FAIL: no matching E2E manifest found for $MODEL"
    echo "Create a manifest at tests/e2e/models/<family>/manifests/<name>.json and list it in tests/e2e/models/<family>/MODEL.toml before declaring success."
    STEPS+=("FAIL  E2E pytest (no manifest -- create one)")
    FAIL=$((FAIL + 1))
elif [[ "$BUILD_SUCCEEDED" != "true" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "FAIL: this invocation did not produce a usable bundle -- refusing to rebuild from manifest hf_id"
    STEPS+=("FAIL  E2E pytest (current bundle unavailable)")
    FAIL=$((FAIL + 1))
elif [[ ! -x "$BINARY" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "FAIL: C++ binary not found at $BINARY -- cannot validate E2E"
    STEPS+=("FAIL  E2E pytest (no binary)")
    FAIL=$((FAIL + 1))
fi

if [[ "$FAIL" -eq 0 ]]; then
    if mv -f -- "$CANDIDATE_BUNDLE_PATH" "$BUNDLE_PATH"; then
        echo ""
        echo "Published validated bundle: $BUNDLE_PATH"
    else
        echo ""
        echo "FAIL: unable to publish validated bundle to $BUNDLE_PATH"
        STEPS+=("FAIL  publish validated bundle")
        FAIL=$((FAIL + 1))
    fi
fi

# Summary
echo ""
echo "========================================"
echo "  Validation Summary: $MODEL"
echo "========================================"
for s in "${STEPS[@]}"; do
    echo "  $s"
done
echo "----------------------------------------"
echo "  $PASS passed, $FAIL failed"
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
