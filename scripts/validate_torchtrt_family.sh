#!/usr/bin/env bash
# Validate a Torch-TRT model family end-to-end:
#   1. Build .trtfb bundle
#   2. Diff logits (Torch-TRT StaticCache vs HF eager)
#   3. Runner parity (Python Torch-TRT vs C++ binary) [if binary available]
#
# Usage:
#   ./scripts/validate_torchtrt_family.sh Qwen/Qwen3-0.6B
#   ./scripts/validate_torchtrt_family.sh Qwen/Qwen3-0.6B --max-cache-length 512
#   ./scripts/validate_torchtrt_family.sh Qwen/Qwen3-0.6B --binary ./build/trtmc
#
# Requirements: torch, torch_tensorrt, tensorrt_model_connect[torch-trt] installed, C++ binary built.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
MAX_CACHE_LENGTH=256
BINARY="${PROJECT_DIR}/build/trtmc"
BUNDLE_DIR="/tmp"
TRUST_REMOTE_CODE=""
PRECISION="fp16"

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-cache-length) MAX_CACHE_LENGTH="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --trust-remote-code) TRUST_REMOTE_CODE="--trust-remote-code"; shift ;;
        --precision) PRECISION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <model-id-or-path> [--max-cache-length N] [--binary PATH] [--bundle-dir DIR] [--precision fp16|bf16|fp32] [--trust-remote-code]"
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
    echo "Usage: $0 <model-id-or-path> [options]" >&2
    exit 1
fi

# Derive a safe bundle filename
SAFE_NAME="$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')"
BUNDLE_PATH="${BUNDLE_DIR}/${SAFE_NAME}.trtfb"

# Python with HF/TRT deps
HF_PYTHON="${HF_PYTHON:-/opt/venv/bin/python}"
if [[ ! -x "$HF_PYTHON" ]]; then
    HF_PYTHON="$(which python3 2>/dev/null || echo python3)"
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
run_step "Build .trtfb bundle" \
    env TRTMC_PYTHON="$HF_PYTHON" "$BINARY" build --method torchtrt "$MODEL" -o "$BUNDLE_PATH" \
        --max-cache-length "$MAX_CACHE_LENGTH" \
        --precision "$PRECISION"

# Step 2: diff_torchtrt (logit comparison: StaticCache vs eager)
run_step "diff_torchtrt (battery)" \
    "$HF_PYTHON" "${PROJECT_DIR}/tools/diff_torchtrt.py" \
        --model "$MODEL" --atol 1e-2 --battery \
        --max-cache-length "$MAX_CACHE_LENGTH" $TRUST_REMOTE_CODE

# Step 3: C++ runner (if binary exists)
if [[ -x "$BINARY" ]]; then
    run_step "C++ inference" \
        "$BINARY" run "$BUNDLE_PATH" \
            --prompt "The capital of France is" \
            --max-new-tokens 10 \
            --hf-python "$HF_PYTHON"
else
    echo ""
    echo "==== C++ inference ===="
    echo "SKIP: C++ binary not found at $BINARY"
    STEPS+=("SKIP  C++ inference (no binary)")
fi

# Summary
echo ""
echo "========================================"
echo "  Torch-TRT Validation Summary: $MODEL"
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
