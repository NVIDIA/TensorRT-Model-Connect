#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ulimit -c 0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/trtmc-issue428-pure-trt}
MODEL=${MODEL:-google/gemma-2-2b-it}
CACHE_LENGTH=${CACHE_LENGTH:-1741}
PYTHON=${PYTHON:-python3}
SKIP_BUILD=0
FORCE_BUILD=0
EXPECT_FIXED=${EXPECT_FIXED:-0}

usage() {
    echo "Usage: $0 [--output-dir PATH] [--skip-build] [--force-build] [--expect-fixed]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --force-build)
            FORCE_BUILD=1
            shift
            ;;
        --expect-fixed)
            EXPECT_FIXED=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p -- "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

PREFIX="gemma-2-2b-c${CACHE_LENGTH}"
PREFILL_PLAN="${OUTPUT_DIR}/${PREFIX}-prefill.plan"
DECODE_PLAN="${OUTPUT_DIR}/${PREFIX}-decode.plan"
BUILD_LOG="${OUTPUT_DIR}/build.log"
INFER_LOG="${OUTPUT_DIR}/infer.log"

echo "[one-click] repo=${REPO_ROOT}"
echo "[one-click] output_dir=${OUTPUT_DIR}"
echo "[one-click] model=${MODEL} cache_length=${CACHE_LENGTH}"

"${PYTHON}" -c 'import tensorrt as trt; print(f"[one-click] TensorRT={trt.__version__}")'
"${PYTHON}" -c 'from cuda.bindings import runtime as c; s,n=c.cudaGetDeviceCount(); print(f"[one-click] cuda_status={s} gpu_count={n}"); raise SystemExit(0 if int(s)==0 and n>0 else 2)'

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    BUILD_ARGS=(
        --model "${MODEL}"
        --output-dir "${OUTPUT_DIR}"
        --cache-length "${CACHE_LENGTH}"
    )
    if [[ ${FORCE_BUILD} -eq 1 ]]; then
        BUILD_ARGS+=(--force)
    fi
    "${PYTHON}" "${SCRIPT_DIR}/build_plans.py" "${BUILD_ARGS[@]}" 2>&1 | tee "${BUILD_LOG}"
fi

if [[ ! -s "${PREFILL_PLAN}" || ! -s "${DECODE_PLAN}" ]]; then
    echo "[one-click] SETUP_ERROR missing plan(s): ${PREFILL_PLAN} ${DECODE_PLAN}" >&2
    exit 2
fi

set +e
"${PYTHON}" "${SCRIPT_DIR}/infer_sequence.py" \
    --prefill-plan "${PREFILL_PLAN}" \
    --decode-plan "${DECODE_PLAN}" \
    --cache-length "${CACHE_LENGTH}" \
    2>&1 | tee "${INFER_LOG}"
INFER_RC=${PIPESTATUS[0]}
set -e

REPRODUCED=0
if grep -Eiq \
    'ISSUE_428_REPRODUCED|CUDA (error|status)[^0-9]*700|illegal memory access|l2cm_layer_internal' \
    "${INFER_LOG}"; then
    REPRODUCED=1
fi

echo "[one-click] infer_rc=${INFER_RC} reproduced=${REPRODUCED} log=${INFER_LOG}"

if [[ ${EXPECT_FIXED} -eq 1 ]]; then
    if [[ ${INFER_RC} -eq 0 && ${REPRODUCED} -eq 0 ]]; then
        echo "[one-click] FIX_VERIFIED"
        exit 0
    fi
    echo "[one-click] FIX_NOT_VERIFIED"
    exit 1
fi

if [[ ${REPRODUCED} -eq 1 ]]; then
    echo "[one-click] ISSUE_428_REPRODUCED"
    exit 0
fi
if [[ ${INFER_RC} -eq 0 ]]; then
    echo "[one-click] ISSUE_428_NOT_REPRODUCED"
    exit 1
fi
echo "[one-click] UNEXPECTED_INFER_FAILURE"
exit 2
