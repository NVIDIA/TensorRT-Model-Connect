---
title: Build from Source
description: Build the CLI, TensorRT backend, and Qwen DSO for one selected GPU.
---

Use this path on Linux x86_64 or aarch64 for the first Qwen inference from
source. Start at the repository root.

## Automated environment preparation

The repository-local `scripts/devToolkit` Python API can inspect the host,
resolve an exact TensorRT/CUDA cohort, build or reuse the development image,
start an owned persistent container, build TRTMC, verify the installation, and
write a reproducibility receipt:

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts" / "devToolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTarget, PrepareRequest

toolkit = DevToolkit.from_checkout(repo)
plan = toolkit.plan(
    PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        target=DockerTarget(gpu="0"),
        mode="development",
    )
)

for step in plan.steps:
    print(step.id, step.description)

result = toolkit.apply(plan)
print(result.environment.activate_command)
print(result.receipt)
```

`plan()` is read-only. `apply()` leaves the labelled container running for
later development and stores state under `.devtoolkit/`. It never installs a
host driver or changes host CUDA/TensorRT libraries. See
`scripts/devToolkit/README.md` for local, installed-wheel, model-smoke, and
downstream handoff examples.

The manual commands below remain the direct source-build path and show the
operations performed by development mode.

## 1. Select the GPU and start the container

Change only `GPU`. The commands derive the SM used by CMake and select the
matching development Dockerfile. Repository CI continues to use `Dockerfile`.

```bash
GPU=0
SM="$(
  nvidia-smi -i "$GPU" \
    --query-gpu=compute_cap \
    --format=csv,noheader,nounits |
  tr -d '.[:space:]'
)"
IMAGE="trtmc-quickstart"

case "$(uname -m)" in
  x86_64) DOCKERFILE=Dockerfile.dev.x86 ;;
  aarch64) DOCKERFILE=Dockerfile.dev.aarch64 ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

docker build \
  -f "$DOCKERFILE" \
  -t "$IMAGE" requirements

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
  -DTRTMC_BUILD_BENCHMARKS=OFF \
  -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF

cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc_backend_trt \
  trtmc_model_qwen

export TRTMC_MODEL_PLUGIN_DIR="$TRTMC_BUILD_DIR/models"
export PATH="$PWD/$TRTMC_BUILD_DIR:$PATH"
```

This path skips CI-only Python profiles and unrelated model DSOs. Continue to
[Quick Start](quick-start.md) in the same container shell. Full-repository and
advanced backend options belong in the
[Build System](../architecture/build-system.md) reference.

{/* Collaborative review anchor: batch 2. */}
