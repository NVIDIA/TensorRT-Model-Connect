---
title: Build from Source
description: Build the CLI, TensorRT backend, and model DSOs for one selected GPU.
---

Use this path on Linux x86_64 or aarch64 when building the native CLI and
runtime from source. Start at the repository root.

## 1. Select the GPU and start the container

Change only `GPU`. The commands derive the compute capability used by Docker
and CMake, then select the matching repository Dockerfile for the host
architecture.

```bash
GPU=0
CC="$(
  nvidia-smi -i "$GPU" \
    --query-gpu=compute_cap \
    --format=csv,noheader,nounits |
  tr -d '[:space:]'
)"
SM="${CC/.}"
IMAGE="trtmc-dev-sm${SM}"

case "$(uname -m)" in
  x86_64) DOCKERFILE=Dockerfile.x86 ;;
  aarch64) DOCKERFILE=Dockerfile ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

docker build \
  -f "$DOCKERFILE" \
  --build-arg TRTMC_TORCH_CUDA_ARCH_LIST="$CC" \
  -t "$IMAGE" .

SOURCE_DIR="$(git rev-parse --show-toplevel)"

docker run --rm -it \
  --gpus "device=${GPU}" \
  --ipc=host \
  --mount "type=bind,source=${SOURCE_DIR},target=/src" \
  --workdir /src \
  --env TRTMC_SM="$SM" \
  "$IMAGE" \
  bash
```

Run the remaining commands inside the container.

## 2. Build the native runtime

```bash
python -m pip install --no-deps -e . -C py-only=true

TRTMC_BUILD_DIR="build-sm${TRTMC_SM}"

cmake -S . -B "$TRTMC_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_SM}-real" \
  -DTRTMC_BUILD_BACKEND_TRT=ON \
  -DTRTMC_BUILD_BACKEND_RTX=OFF \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_BENCHMARKS=OFF

cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc_backend_trt \
  trtmc_model_plugins

export TRTMC_MODEL_PLUGIN_DIR="$TRTMC_BUILD_DIR/models"
export PATH="$PWD/$TRTMC_BUILD_DIR:$PATH"
```

Continue to [Quick Start](quick-start.md) in the same container shell. Advanced
backend and focused-target options belong in the
[Build System](../architecture/build-system.md) reference.

{/* Collaborative review anchor. */}
