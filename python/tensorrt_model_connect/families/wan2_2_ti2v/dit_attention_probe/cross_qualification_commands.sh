#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reproduce the isolated Wan2.2 block-0 cross-SDPA qualification.
set -euo pipefail

ROOT=/workspace/tensorrt-model-connect
PROBE=${ROOT}/python/tensorrt_model_connect/families/wan2_2_ti2v/dit_attention_probe
ARTIFACT=/workspace/results/wan2_2_ti2v_5b/dit_attention_probe
FRONTEND=/tmp/cudnn-frontend-1.22.1-wan22

if [[ ! -d "${FRONTEND}/include" ]]; then
  git clone --depth 1 --branch v1.22.1 https://github.com/NVIDIA/cudnn-frontend.git "${FRONTEND}"
fi

cmake -S "${PROBE}" -B "${ARTIFACT}/build" \
  -DWAN22_SDPA_CUDNN_FRONTEND_INCLUDE_DIR="${FRONTEND}/include" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${ARTIFACT}/build" --parallel 8

CUDA_VISIBLE_DEVICES=2 \
CUDNN_FRONTEND_LOG_INFO=1 \
CUDNN_FRONTEND_LOG_FILE="${ARTIFACT}/cross/native/cudnn_frontend_build.log" \
/opt/venv/bin/python "${PROBE}/build_trt_probe.py" \
  --plugin "${ARTIFACT}/build/libtrtmc_wan22_cudnn_sdpa_probe.so" \
  --output "${ARTIFACT}/cross/native/wan22_block0_cross_sdpa.plan" \
  --report "${ARTIFACT}/cross/native/build_report.json" \
  --q-sequence 27280 \
  --kv-sequence 512 \
  --engine-id 10 \
  --kernel-config 7

CUDA_VISIBLE_DEVICES=2 \
CUDNN_FRONTEND_LOG_INFO=1 \
CUDNN_FRONTEND_LOG_FILE="${ARTIFACT}/cross/native/cudnn_frontend_run.log" \
"${ARTIFACT}/build/wan22_cudnn_sdpa_trt_runner" \
  "${ARTIFACT}/build/libtrtmc_wan22_cudnn_sdpa_probe.so" \
  "${ARTIFACT}/cross/native/wan22_block0_cross_sdpa.plan" \
  "${ARTIFACT}/cross/capture/q_bshd_bf16.bin" \
  "${ARTIFACT}/cross/capture/k_bshd_bf16.bin" \
  "${ARTIFACT}/cross/capture/v_bshd_bf16.bin" \
  "${ARTIFACT}/cross/native/o_trt_cpp_bshd_bf16.bin" \
  1 \
  5

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python "${PROBE}/qualify_native_sdpa.py" \
  --capture-manifest "${ARTIFACT}/cross/capture/manifest.json" \
  --actual "${ARTIFACT}/cross/native/o_trt_cpp_bshd_bf16.bin" \
  --output "${ARTIFACT}/cross/native/qualification.json" \
  --implementation TensorRT_IPluginV2DynamicExt_cuDNN_Graph \
  --engine-config eng10_k24=7

ldd "${ARTIFACT}/build/libtrtmc_wan22_cudnn_sdpa_probe.so"
