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

The toolkit reuses a container only when its checkout-owned configuration still
matches. A foreign name collision or configuration drift fails without removing
or replacing the container. Unknown host architectures fail before Docker is
invoked. The toolkit has no environment catalog or secondary artifact identity.
Each development Dockerfile's first `FROM` is its base-image pin; repository CI
continues to use the root `Dockerfile`. Optional Python dependencies remain in
`families/<family>/requirements.txt`. See `apps/devtoolkit/README.md` for
lifecycle policies and the explicit existing-interpreter local path.

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
  -DTRTMC_BUILD_BACKEND_RTX=OFF \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_EXAMPLES=OFF

cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc_backend_trt \
  trtmc_model_qwen

export PATH="$PWD/$TRTMC_BUILD_DIR:$PATH"
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

This path skips CI-only Python profiles and unrelated model DSOs. Continue to
[Quick Start](quick-start.md) in the same container shell. Full-repository
ownership and backend boundaries are documented in the
[AI-Native Horizontal Scaling Architecture](../architecture/ai-native-horizontal-scaling.md).

{/* Collaborative review anchor: batch 2. */}
