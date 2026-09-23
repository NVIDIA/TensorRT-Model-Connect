---
title: Build from Source
description: Build the CLI, TensorRT backend, and Qwen DSO for one selected GPU.
---

Use this path on Linux x86_64 or aarch64 for the first Qwen inference from
source. Start at the repository root.

## Automated environment preparation

The repository-local `apps/devtoolkit` Python API selects
`Dockerfile.dev.x86` or `Dockerfile.dev.aarch64` from the host architecture,
runs the direct development-image build, starts a persistent container for the
current checkout, and optionally installs one family's declared dependencies:

```python
from pathlib import Path
import subprocess
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "apps" / "devtoolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTargetPolicy

gpu = "0"
sm = subprocess.run(
    [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().replace(".", "")

toolkit = DevToolkit.from_checkout(repo)
environment = toolkit.prepare_docker(
    family="qwen",
    gpu=gpu,
    environment={"TRTMC_SM": sm},
    policy=DockerTargetPolicy.ENSURE,
)
print(" ".join(environment.command("bash")))
```

Runnable end-to-end examples are available for both supported execution paths:

```bash
# Build in a checkout-owned development container.
python3 apps/devtoolkit/examples/docker_build.py --gpu 0

# Build directly on a prepared host interpreter/toolchain.
python3 apps/devtoolkit/examples/local_build.py \
  --python /path/to/python3.12 \
  --tensorrt 11.0.0.114
```

The local path expects the generic native build prerequisites documented in
`apps/devtoolkit/README.md`; use its `--cmake-python` and
`--cmake-prefix-path` options when those dependencies live in isolated
prefixes.

The toolkit reuses a container only when its checkout-owned configuration still
matches. A foreign name collision or configuration drift fails without removing
or replacing the container. Unknown host architectures fail before Docker is
invoked. This preparation call remains separate from the toolkit's optional
`resolve`, `provision`, `build`, and `run` capabilities; use
`environment.execution_target()` to pass the prepared container into that
evidence-producing path. Each development Dockerfile's first `FROM` is its
base-image pin; repository CI continues to use the root `Dockerfile`. Optional
Python dependencies remain in `families/<family>/requirements.txt`. See
`apps/devtoolkit/README.md` for lifecycle policies, immutable environment
identity and receipts, managed toolchain catalogs, and the explicit
existing-interpreter local path.

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

The development image already contains the builder's Python dependencies. Put
the checkout's builder and family packages on `PYTHONPATH` instead of installing
the project with pip; a pip install of this checkout compiles every family DSO.

```bash
export PYTHONPATH="$PWD/core/builder:$PWD${PYTHONPATH:+:$PYTHONPATH}"

TRTMC_BUILD_DIR="build-sm${TRTMC_SM}"

cmake -S . -B "$TRTMC_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_SM}-real" \
  -DTRTMC_BUILD_BACKEND_RTX=OFF \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_EXAMPLES=OFF

cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc-server \
  trtmc_backend_trt \
  trtmc_model_qwen

export PATH="$PWD/$TRTMC_BUILD_DIR:$PATH"
```

To run the optional text server from this build, install only its Python
control plane dependencies. The source-built `trtmc-server` finds its control
plane package beside the executable:

```bash
python -m pip install 'fastapi>=0.115,<0.142' 'pydantic>=2.11,<3' 'uvicorn>=0.30,<0.53'
```

TensorRT-RTX is an explicit optional build. When its SDK is installed, enable
only its backend DSO with the exact include and library directories:

```bash
cmake -S . -B "$TRTMC_BUILD_DIR" \
  -DTRTMC_BUILD_BACKEND_RTX=ON \
  -DTRTMC_RTX_INCLUDE_DIR=/absolute/tensorrt-rtx/include \
  -DTRTMC_RTX_LIBRARY_DIR=/absolute/tensorrt-rtx/lib
cmake --build "$TRTMC_BUILD_DIR" --target trtmc_backend_rtx
```

## 3. Build and run Qwen

This build contains only the Qwen family DSO, so check it with a Qwen
checkpoint. The build directory is the runtime root:

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --precision fp16 \
  --output qwen3-0.6b.bundle

trtmc run qwen3-0.6b.bundle \
  --runtime-root "$PWD/$TRTMC_BUILD_DIR" \
  --prompt "Explain what TensorRT does in one sentence." \
  --max-new-tokens 64
```

This path skips CI-only Python profiles and unrelated model DSOs. To run
another family, add its `trtmc_model_<family>` target to the build above, then
continue to [Quick Start](quick-start.md) in the same container shell. Full-repository
ownership and backend boundaries are documented in the
[AI-Native Horizontal Scaling Architecture](../architecture/ai-native-horizontal-scaling.md).

{/* Collaborative review anchor: batch 2. */}
