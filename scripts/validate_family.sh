#!/usr/bin/env bash
# Validate a model family end-to-end: build bundle, diff logits, diff layers, runner parity.
#
# Usage:
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B                       # HF repo ID
#   ./scripts/validate_family.sh models/hf/Qwen__Qwen3-0.6B            # local dir
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B --max-cache-length 512
#   ./scripts/validate_family.sh Qwen/Qwen3-0.6B --binary ./build/trtmc
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

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cache-length) MAX_CACHE_LENGTH="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --trust-remote-code) TRUST_REMOTE_CODE="--trust-remote-code"; shift ;;
        -h|--help)
            echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--trust-remote-code]"
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
DECODER_STRATEGIES="decoder_kv_cache decoder_moe ssm_recurrent rwkv_recurrent hybrid_mamba_attention"
IS_DECODER=false
for s in $DECODER_STRATEGIES; do
    if [[ "$RUNTIME_STRATEGY" == "$s" ]]; then
        IS_DECODER=true
        break
    fi
done

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
# Find manifest by matching hf_id or model name in tests/e2e/models/*.json
E2E_MODEL=""
for manifest in "${PROJECT_DIR}"/tests/e2e/models/*.json; do
    hf_id=$("$HF_PYTHON" -c "import json; d=json.load(open('$manifest')); print(d.get('hf_id',''))" 2>/dev/null || true)
    manifest_name=$("$HF_PYTHON" -c "import json; d=json.load(open('$manifest')); print(d.get('name',''))" 2>/dev/null || true)
    skip=$("$HF_PYTHON" -c "import json; d=json.load(open('$manifest')); print(d.get('skip',''))" 2>/dev/null || true)
    if [[ -n "$skip" ]]; then
        continue
    fi
    if [[ "$hf_id" == "$MODEL" ]] || [[ "$MODEL" == *"$manifest_name"* ]]; then
        E2E_MODEL="$manifest_name"
        break
    fi
done

if [[ -n "$E2E_MODEL" ]] && [[ -x "$BINARY" ]]; then
    ENGINE_DIR="${ENGINE_DIR:-/workspace/users/yifeif/tensorrt-model-connect/engines}"
    run_step "E2E pytest [${E2E_MODEL}]" \
        "$HF_PYTHON" -m pytest "${PROJECT_DIR}/tests/test_e2e.py::test_e2e[${E2E_MODEL}]" -v \
            --engine-dir "$ENGINE_DIR" \
            --trtmc-binary "$BINARY" --hf-python "$HF_PYTHON" \
            --rebuild-engines
elif [[ -z "$E2E_MODEL" ]]; then
    echo ""
    echo "==== E2E pytest ===="
    echo "WARN: no matching E2E manifest found for $MODEL"
    echo "Create a manifest at tests/e2e/models/<name>.json before declaring success."
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
