---
title: Build from Source
description: Build the development image, CLI, TensorRT backend, and native model runtime DSOs for one target SM.
---

Use this path when you are developing the native CLI, a backend, or one or more
model runtime DSOs. If you only need an installable release artifact, use the
wheel paths in [Installation](installation.md) instead.

Start with the repository already cloned and a terminal open at its root. The
host needs a working NVIDIA driver, Docker Engine with NVIDIA Container Toolkit
support, network access, and enough disk space to build the image and model
bundles. No other host path is assumed.

## 1. Select one target SM

The development image is model-agnostic: it supplies TensorRT, CUDA, Python,
CMake, Ninja, and the declared Python execution profiles. Select the compute
capability needed by the target system rather than a model family.

`TRTMC_TORCH_SM` uses the dotted compute-capability form required by
`TORCH_CUDA_ARCH_LIST`. `TRTMC_SM` uses the integer form required by CMake. For
example, build an image for SM103 with:

```bash
TRTMC_SM=103
TRTMC_TORCH_SM=10.3
TRTMC_IMAGE="trtmc-dev-sm${TRTMC_SM}"

docker build \
  --build-arg TRTMC_TORCH_CUDA_ARCH_LIST="$TRTMC_TORCH_SM" \
  --tag "$TRTMC_IMAGE" .
```

For SM110, use the corresponding pair:

```bash
TRTMC_SM=110
TRTMC_TORCH_SM=11.0
TRTMC_IMAGE="trtmc-dev-sm${TRTMC_SM}"

docker build \
  --build-arg TRTMC_TORCH_CUDA_ARCH_LIST="$TRTMC_TORCH_SM" \
  --tag "$TRTMC_IMAGE" .
```

The Docker build argument sets the architecture list used when Python CUDA
extensions are compiled in the image or later in the container. The CMake
configuration controls native CUDA code in TRTMC DSOs. Keep the two values
aligned to the same compute capability.

When a matching prebuilt development image is published, `docker pull` can
replace `docker build`. The remaining source-mount and CMake steps are the
same.

## 2. Start the development container

Choose the image tag created above. The source path is resolved from Git, and
`/src` is a container-local mount point created by Docker; neither path needs
to be known in advance.

```bash
TRTMC_SM=103
TRTMC_IMAGE="trtmc-dev-sm${TRTMC_SM}"
TRTMC_SOURCE_DIR="$(git rev-parse --show-toplevel)"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --mount "type=bind,source=$TRTMC_SOURCE_DIR,target=/src" \
  --workdir /src \
  "$TRTMC_IMAGE" bash
```

Run the remaining commands inside this container. Re-enter the same integer SM
selected for the image, then install the editable Python builder once:

```bash
TRTMC_SM=103
python -m pip install --no-deps -e . -C py-only=true
```

The editable install does not compile the native CLI or DSOs. Choose one of the
native build configurations below for that work.

## 3. Choose a native build configuration

### Default: standard TensorRT backend and every model DSO

**Use case:** first-time source setup and general development across the model
inventory. This is the recommended configuration before following the Qwen
Quick Start.

```bash
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
"$TRTMC_BUILD_DIR/trtmc" version
```

The aggregate `trtmc_model_plugins` target builds every manifest-declared model
DSO. Their native CUDA code contains only the selected SM.

### Focused: one model DSO

**Use case:** iterating on one runtime family when rebuilding every model DSO
would add unnecessary compile time. The target name uses the runtime owner ID,
not a Hugging Face repository ID or internal test profile. Qwen models, for
example, use the `qwen` owner.

```bash
TRTMC_MODEL=qwen
TRTMC_BUILD_DIR="build-sm${TRTMC_SM}-${TRTMC_MODEL}"

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
  "trtmc_model_${TRTMC_MODEL}"

export TRTMC_MODEL_PLUGIN_DIR="$TRTMC_BUILD_DIR/models"
"$TRTMC_BUILD_DIR/trtmc" version
```

Runtime owner IDs are directory names under `src/runtime/models/`. Each
directory's `MODEL.toml` declares its output DSO and runtime strategies.

### Optional: TensorRT-RTX backend

**Use case:** building and running native bundles explicitly created with
`trtmc build --rtx`. Do not use this configuration for a standard TensorRT
bundle.

The generic development image includes the standard TensorRT SDK, not the
optional TensorRT-RTX SDK. Install the matching `tensorrt_rtx` Python package,
headers, and `libtensorrt_rtx.so` before configuring this variant. Set the SDK
directories explicitly; no installation path is assumed.

```bash
: "${TRTMC_RTX_INCLUDE_DIR:?Set this to the TensorRT-RTX include directory}"
: "${TRTMC_RTX_LIBRARY_DIR:?Set this to the directory containing libtensorrt_rtx.so}"
python -c "import tensorrt_rtx"

TRTMC_BUILD_DIR="build-sm${TRTMC_SM}-rtx"
cmake -S . -B "$TRTMC_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_SM}-real" \
  -DTRTMC_BUILD_BACKEND_TRT=OFF \
  -DTRTMC_BUILD_BACKEND_RTX=ON \
  -DTRTMC_RTX_INCLUDE_DIR="$TRTMC_RTX_INCLUDE_DIR" \
  -DTRTMC_RTX_LIBRARY_DIR="$TRTMC_RTX_LIBRARY_DIR" \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_BENCHMARKS=OFF
cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc_backend_rtx \
  trtmc_model_plugins
```

This emits `libtrtmc_backend_trt_rtx.so`. The Python package is also required
when creating the matching bundle:

```bash
export TRTMC_MODEL_PLUGIN_DIR="$TRTMC_BUILD_DIR/models"
"$TRTMC_BUILD_DIR/trtmc" build Qwen/Qwen3-0.6B \
  --rtx \
  --output qwen3-0.6b-rtx.bundle
```

## 4. Run the first inference

After the default build, keep `TRTMC_MODEL_PLUGIN_DIR` exported and use the
source-built executable for the canonical first-inference workflow:

```bash
export TRTMC="./build-sm${TRTMC_SM}/trtmc"
```

Continue with [Your First NLP Inference](quick-start.md), using `$TRTMC` for
the build, inspect, and run commands. The [Build System](../architecture/build-system.md)
reference explains the CLI, backend, and model target boundaries in more
detail.

{/* Collaborative review anchor. */}
