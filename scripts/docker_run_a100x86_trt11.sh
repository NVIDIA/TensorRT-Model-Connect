#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${TRTMC_DOCKER_IMAGE:-trtmc-dev-a100x86-trt11}"
CONTAINER_NAME="${TRTMC_DOCKER_CONTAINER:-trtmc-dev-a100x86-trt11}"
STORAGE_ROOT="${TRTMC_STORAGE_ROOT:-$HOME/trtmc-storage}"
HF_CACHE="${TRTMC_HF_CACHE:-$HOME/.cache/huggingface/hub}"
ENGINE_DIR="${STORAGE_ROOT}/engines"
TRT11_ROOT="${TRTMC_TRT11_ROOT:-/tmp/trt11-install/external/TensorRT-11.0.0.107}"
NCCL_ROOT="${TRTMC_NCCL_ROOT:-/tmp/trt11-install/external/nccl_2.30.4-1+cuda13.2_x86_64}"

for required_dir in "$TRT11_ROOT" "$NCCL_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "Required directory is missing: $required_dir" >&2
    exit 1
  fi
done

mkdir -p "$HF_CACHE" "$ENGINE_DIR" 2>/dev/null || true

docker_args=(--rm)
if [ -t 0 ] && [ -t 1 ]; then
  docker_args+=(-it)
fi

docker run "${docker_args[@]}" \
  --gpus all \
  -v "$PWD":/workspace/tensorrt-model-connect \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -v "${TRT11_ROOT}":/opt/tensorrt-11:ro \
  -v "${NCCL_ROOT}":/opt/nccl-2.30:ro \
  -e ENGINE_DIR="${ENGINE_DIR}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e TRT11_ROOT=/opt/tensorrt-11 \
  -e NCCL_ROOT=/opt/nccl-2.30 \
  -e TRTMC_TRT_INCLUDE_DIR=/opt/tensorrt-11/include \
  -e TRTMC_TRT_LIBRARY=/opt/tensorrt-11/lib/libnvinfer.so \
  -e TRT_INC_DIR=/opt/tensorrt-11/include \
  -e TRT_LIB_DIR=/opt/tensorrt-11/lib \
  -e TRTMC_NCCL_LIB_DIR=/opt/nccl-2.30/lib \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e LD_PRELOAD= \
  -e LD_LIBRARY_PATH=/opt/nccl-2.30/lib:/opt/tensorrt-11/lib:/usr/local/cuda/lib64:/opt/venv/lib/python3.12/site-packages/tensorrt_libs \
  -w /workspace/tensorrt-model-connect \
  --name "$CONTAINER_NAME" \
  "$IMAGE" ./scripts/bootstrap_trt11_container.sh "$@"
