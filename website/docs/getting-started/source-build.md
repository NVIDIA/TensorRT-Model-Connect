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

YOLOv5, YOLOv8, YOLOv10, and YOLO11 builds do not require the `ultralytics`
package. Supply your own appropriately licensed checkpoint; Model Connect does
not include model weights. The `.pt` readers require PyTorch and accept only
trusted checkpoints, because the archive format uses pickle.

Their accuracy references live under each family's `tests/reference/` and use
ordinary PyTorch layers with torchvision suppression where applicable. Neither
builds nor tests install or import the Ultralytics package. E2E references run
with the normal test interpreter; standalone qualification declares its optional
dependencies in `tests/requirements.txt`. The reference implementations and their
dependency declarations are excluded from the wheel. Checkpoint identifiers and
serialized class names still identify the upstream model format. The references
use the same checkpoints and preserve the existing accuracy thresholds. Selected
E2E tests also check both the original and mirrored image against numeric
snapshots recorded before the upstream package was removed. The snapshots pin
checkpoint and image hashes; they contain predictions, not model weights.

The manual commands below remain the direct source-build path and show the
operations performed by development mode.

## Select native families

CMake configures all discovered families by default. To configure one family,
pass `-DTRTMC_BUILD_FAMILIES=qwen`; for several families, pass a quoted
semicolon-separated list. An empty list, `-DTRTMC_BUILD_FAMILIES=""`, configures
only the shared runtime, backends, applications, and their tests. This lets a
CLI/backend-only build avoid unselected families' optional native prerequisites.
Unknown family names fail configuration. Omitting the option or setting it to
`*` preserves the default of configuring every family.

## Optional Edge-LLM SDK

Edge is off by default. Set `-DTRTMC_ENABLE_EDGELLM=ON` to select the
0.12.0 C++ SDK. To build against 0.11.0 instead, also pass
`-DTRTMC_EDGELLM_RELEASE=0.11.0` in a separate build directory. Keep the 0.12 Python
builder in a separate environment: its Transformers requirement is different
from Model Connect's shared pin. This option provisions the native SDK; it
does not port any family adapter. See the [native SDK setup](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/cmake/edge_llm/README.md)
for provisioning options. Follow each family's instructions for model builds.

To include the optional GR00T and Pi0.5 C++ libraries and headers in the 0.12
SDK, add `-DTRTMC_EDGELLM_POLICY_SDK=ON` when provisioning from source, or
select a matching external SDK that already contains both. The base SDK does
not contain them; installing the Edge Python wheel does not add them.

Build on the GPU where the SDK will run. Pass its numeric architecture:
for example, `-DCMAKE_CUDA_ARCHITECTURES=100` on GB200 or
`-DCMAKE_CUDA_ARCHITECTURES=103` on GB300. The 0.12 provider selects the
matching accelerated compiler target (`100a` or `103a`) for both the SDK
and its consumers. The device check and installed receipt remain numeric.
This compiler selection does not qualify a model on that GPU.

## Optional Edge-LLM Python tools

The `edgellm` extra installs the 0.12.0 **base** package in the Model Connect
environment. It does not install the builder dependencies or C++ SDK and does
not select a family backend. After preparing the native build prerequisites:

```bash
python -m pip install -e '.[edgellm]'
python -c 'from importlib.metadata import version; print(version("tensorrt-edgellm"))'
```

Install the builder in a separate environment. Match its TensorRT Python
version to the native SDK; for example:

```bash
python3 -m venv "$HOME/.venvs/edge012"
"$HOME/.venvs/edge012/bin/python" -m pip install 'tensorrt==11.4.0.106' 'tensorrt-edgellm[builder]==0.12.0'
```

Alternatively, install `.[builder]` from the official 0.12.0 source checkout
in that environment. The builder requires Transformers 5.17.0; Model Connect
uses 5.2.0. Do not override either pin. Select the builder using the owning
family's documented option. The wheel does not replace the native CUDA,
TensorRT, or Edge SDK requirements.

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
  trtmc-server \
  trtmc_backend_trt \
  trtmc_model_qwen

export PATH="$PWD/$TRTMC_BUILD_DIR:$PATH"
```

The command above builds the Qwen family. If you configured with
`-DTRTMC_BUILD_FAMILIES=""`, omit `trtmc_model_qwen` from the target list. For a
specific family selection, replace it with the corresponding selected targets,
such as `trtmc_model_qwen2_5_omni` for `-DTRTMC_BUILD_FAMILIES=qwen2_5_omni`.

To run the optional text server from this build, install its Python control
plane dependencies with `python -m pip install -e '.[serve]' -C py-only=true`.

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
