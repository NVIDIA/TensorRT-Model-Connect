#!/usr/bin/env bash
# Validate a model family end-to-end: build bundle, diff logits, diff layers, runner parity.
#
# Usage:
#   ./scripts/validate_family.sh org/example-decoder                  # HF repo ID
#   ./scripts/validate_family.sh models/hf/org__example-decoder       # local dir
#   ./scripts/validate_family.sh org/example-decoder --max-cache-length 512
#   ./scripts/validate_family.sh org/example-decoder --binary ./build/trtmc
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
TRUST_REMOTE_CODE=""
MODEL_PLUGIN_DIR=""
ISOLATE_MODEL_PLUGIN="false"

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cache-length) MAX_CACHE_LENGTH="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --model-plugin-dir) MODEL_PLUGIN_DIR="$2"; shift 2 ;;
        --isolate-model-plugin) ISOLATE_MODEL_PLUGIN="true"; shift ;;
        --trust-remote-code) TRUST_REMOTE_CODE="--trust-remote-code"; shift ;;
        -h|--help)
            echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--model-plugin-dir DIR] [--isolate-model-plugin] [--trust-remote-code]"
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
    echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH]" >&2
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

# Derive a safe bundle filename from the model ID.
SAFE_NAME="$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')"
BUNDLE_PATH="${BUNDLE_DIR}/${SAFE_NAME}.trtfb"

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

# Step 1: Build bundle
run_step "Build bundle" \
    "$BINARY" build "$MODEL" -o "$BUNDLE_PATH" \
        --max-cache-length "$MAX_CACHE_LENGTH"

# Detect runtime strategy from the built bundle to skip decoder-only tools
# for encoder-only / seq2seq models (diff_logits, diff_layers, parity only
# work with decoder models that use TrtRunner).
if [[ -x "$BINARY" ]]; then
    RUNTIME_STRATEGY=$("$BINARY" inspect "$BUNDLE_PATH" 2>/dev/null \
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
            --max-cache-length "$MAX_CACHE_LENGTH" $TRUST_REMOTE_CODE

    run_step "diff_layers" \
        "$HF_PYTHON" "${PROJECT_DIR}/tools/diff_layers.py" \
            --model "$MODEL" --atol 0.05 \
            --max-cache-length "$MAX_CACHE_LENGTH" $TRUST_REMOTE_CODE

    if [[ -x "$BINARY" ]]; then
        run_step "test_runner_parity" \
            "$HF_PYTHON" "${PROJECT_DIR}/tools/test_runner_parity.py" \
                --bundle "$BUNDLE_PATH" --binary "$BINARY" \
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

# Step 5: E2E pytest (if a manifest exists for this model)
# Find manifest by matching hf_id or model name in supported E2E manifest layouts.
E2E_MODEL=""
E2E_FAMILY=""
while IFS= read -r -d '' manifest; do
    hf_id=$("$HF_PYTHON" -c 'import json, sys; d=json.load(open(sys.argv[1])); print(d.get("hf_id",""))' "$manifest" 2>/dev/null || true)
    manifest_name=$("$HF_PYTHON" -c 'import json, sys; d=json.load(open(sys.argv[1])); print(d.get("name",""))' "$manifest" 2>/dev/null || true)
    skip=$("$HF_PYTHON" -c 'import json, sys; d=json.load(open(sys.argv[1])); print(d.get("skip",""))' "$manifest" 2>/dev/null || true)
    if [[ -n "$skip" ]]; then
        continue
    fi
    if [[ "$hf_id" == "$MODEL" ]] || [[ "$MODEL" == *"$manifest_name"* ]]; then
        E2E_MODEL="$manifest_name"
        E2E_FAMILY="$(basename "$(dirname "$(dirname "$manifest")")")"
        break
    fi
done < <(find "${PROJECT_DIR}/tests/e2e/models" -maxdepth 3 -type f -name "*.json" -print0 | sort -z)

if [[ -n "$E2E_MODEL" ]] && [[ -x "$BINARY" ]]; then
    ENGINE_DIR="${ENGINE_DIR:-/workspace/users/yifeif/tensorrt-model-connect/engines}"
    E2E_NODE="${PROJECT_DIR}/tests/e2e/models/${E2E_FAMILY}/test_${E2E_FAMILY}_e2e.py::test_model_e2e[${E2E_MODEL}]"
    E2E_ARGS=(
        "$E2E_NODE"
        -v
        --engine-dir "$ENGINE_DIR"
        --trtmc-binary "$BINARY"
        --hf-python "$HF_PYTHON"
        --rebuild-engines
    )
    if [[ -n "$MODEL_PLUGIN_DIR" ]]; then
        E2E_ARGS+=(--model-plugin-dir "$MODEL_PLUGIN_DIR")
    fi
    run_step "E2E pytest [${E2E_MODEL}]" \
        "$HF_PYTHON" -m pytest "${E2E_ARGS[@]}"
elif [[ -z "$E2E_MODEL" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "WARN: no matching E2E manifest found for $MODEL"
    echo "Create a manifest at tests/e2e/models/<family>/manifests/<name>.json and list it in tests/e2e/models/<family>/MODEL.toml before declaring success."
    STEPS+=("WARN  E2E pytest (no manifest -- create one)")
elif [[ ! -x "$BINARY" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "FAIL: C++ binary not found at $BINARY -- cannot validate E2E"
    STEPS+=("FAIL  E2E pytest (no binary)")
    FAIL=$((FAIL + 1))
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
