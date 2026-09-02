---
title: Build from Source
description: Build the CLI, TensorRT backend, and Qwen DSO for one selected GPU.
---

Use this path on Linux x86_64 or aarch64 for the first Qwen inference from
source. Start at the repository root.

## Automated environment preparation

The repository-local `scripts/devToolkit` Python API exposes independent
resolution, provisioning, source-build, and command capabilities. TensorRT is
an arbitrary exact four-part request; a cohort is optional qualification
provenance, not an allowlist. This example adopts an existing development
container and verifies its actual CUDA/TensorRT toolchain before building:

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts" / "devToolkit"))

from trtmc_devtoolkit import BuildSpec, DevToolkit, EnvironmentRequest, ExecutionTarget

toolkit = DevToolkit.from_checkout(repo)
lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.2.0.113",
        target=ExecutionTarget.docker(
            container="trtmc-dev-gb300",
            docker_context="default",
            workspace="/workspace/TensorRT-Model-Connect",
        ),
    )
)
environment = toolkit.provision(lock)
build = toolkit.build(
    environment,
    BuildSpec(targets=("trtmc", "trtmc_backend_trt", "trtmc_model_qwen")),
)
print(environment.receipt)
print(build.receipt)
```

The Docker environment lock records the daemon, immutable container, and image
identities and rechecks them before later execution. A reused container name or
changed Docker context therefore fails closed instead of silently selecting a
different environment. Docker CLI 20.10 or newer is required.

`resolve()` is read-only. With CUDA omitted, it prefers a complete target CUDA
toolkit and otherwise selects the managed CUDA 13.3 policy. Managed provisioning
requires digest-pinned artifacts; it never downloads an unverified version.
`provision()` attests Python, CUDA, TensorRT Python/native/header versions and
writes the evidence under `.devtoolkit/`. See `scripts/devToolkit/README.md` for
local targets, explicit CUDA policies, generic TRTMC CLI calls, extension
providers, and the legacy `plan()`/`apply()` compatibility recipe.

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
