# TensorRT-Model-Connect

> Reference implementations for deploying diverse model families on TensorRT.
> Build from a supported checkpoint, run through task-oriented C++ APIs, and
> use the family-owned implementation as a blueprint for your own changes.

Build a deployment bundle from its canonical Hugging Face model ID, then run
native inference in two commands:

```bash
trtmc build Qwen/Qwen3-0.6B --output qwen3-0.6b.bundle
trtmc run qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

### AI-native quick start

Give an AI coding agent with terminal, Docker, and NVIDIA GPU access this
prompt:

```text
/goal Clone https://github.com/NVIDIA/TensorRT-Model-Connect.git into a new TensorRT-Model-Connect directory in the current workspace. Detect the current GPU compute capability, modify the repository development Docker image, build and start the container, install TensorRT-Model-Connect, compile the CLI, TensorRT backend, and all native model DSOs only for that SM, then build and run an end-to-end Qwen/Qwen3-0.6B smoke test. Do not commit or push changes. Report the result of the test, show exact command, input and output of the inference run.
```

Set up the development environment first if `trtmc` is not installed; see
[Installation and setup](#installation-and-setup).

![TensorRT-Model-Connect build and runtime map](website/static/img/diagrams/trtmc-system-map.svg)

[Documentation](https://nvidia.github.io/TensorRT-Model-Connect/) |
[Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start) |
[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview) |
[API Reference](https://nvidia.github.io/TensorRT-Model-Connect/api/overview)

## What is TensorRT-Model-Connect?

TensorRT-Model-Connect (TRTMC) provides a common framework and a growing set of
family-owned reference implementations for running different model families
on TensorRT. Each implementation shows how to turn a supported Hugging Face or
local checkpoint into a `.bundle` bundle and invoke it through task-oriented C++
APIs. The implementations are intended both for straightforward deployment
and as blueprints that developers can inspect, modify, extend, and customize.

Python owns checkpoint resolution and TensorRT engine construction at build
time. Native profiles execute model inference in C++ without PyTorch. A small
number of hybrid profiles explicitly invoke a helper Python executable; their
E2E manifests declare that runtime dependency.

The `.bundle` bundle is the handoff between those environments. Native bundles
resolve their matching model and TensorRT backend DSOs at runtime; exactly
qualified optimized-runtime bundles can carry their implementation DSO. Both
forms still require a compatible NVIDIA driver, CUDA/TensorRT cohort, dynamic
loader, and system libraries.

There is no intermediate ONNX export step. Applications load a bundle and call
task APIs such as `generate()`, `transcribe()`, `generate_image()`,
`embed()`, or `solve()` instead of maintaining conversion stages and
model-specific application glue.

## How does it fit into the TensorRT ecosystem?

Choose the import path that matches the model boundary you already own. Each
path targets TensorRT execution, but the development and deployment interfaces
are different.

| Starting point | Interface | When to use it |
| --- | --- | --- |
| Hugging Face or local checkpoint | **TensorRT-Model-Connect** | Start from a model-family reference implementation, build a `.bundle` bundle for native C++ task inference, and customize the implementation as needed. |
| PyTorch model | **Torch-TensorRT** | Keep the model in the PyTorch ecosystem while compiling its execution with TensorRT. |
| Portable framework interchange | **ONNX** | Use an exchange format when portability across originating frameworks is the primary requirement. |

## Who is this for?

TRTMC is designed for teams that:

- want a working TensorRT reference implementation for a supported model
  instead of starting its builder and runtime integration from scratch;
- need inference in a C++ service, embedded application, robotics stack, or
  edge system and want a concrete deployment blueprint to adapt;
- want to study or customize model-specific builders, native runtime pipelines,
  helper kernels, and integration boundaries;
- want one versioned bundle boundary between a Python-first build environment
  and a native application; or
- need a common task API across text, vision, audio, diffusion, segmentation,
  time-series, and other model families.

TRTMC may not be the right boundary when inference already lives entirely in a
Python/PyTorch deployment, or when ONNX is a required interchange artifact.

## What problems does it solve?

A conventional model-to-deployment path can accumulate several conversion and
integration boundaries:

```text
PyTorch → ONNX or TorchScript → TensorRT → model-specific C++ integration
```

TRTMC reduces that path to:

```text
Hugging Face checkpoint → .bundle artifact → native C++ task API
```

| Traditional pain point | TRTMC boundary |
| --- | --- |
| ONNX export failures and unsupported conversion gaps | Family-owned builders compile supported checkpoints directly with TensorRT APIs. |
| Repeated model-specific application integration | Applications load a bundle and use a task-oriented runtime API. |
| Validation across several conversion artifacts | Build, runtime, and E2E manifests identify one bundle contract and its evidence. |
| Python framework dependencies in native inference paths | Native profiles execute model inference in C++; manifests explicitly flag hybrid profiles that require helper Python. |
| Opaque deployment artifacts | `trtmc inspect` exposes bundle kind, model family, precision, runtime identity, and engines. |

## Broad model coverage

The current repository snapshot declares 78 Python family plugins, 79
unique native runtime strategy keys, and 205 E2E model manifests. These are
discovery counts, not a claim that every model is qualified on every platform.
See [Model Support](website/docs/models-recipes/overview.md) for evidence
levels and the generated inventory.

The build-and-run design spans decoder and hybrid language models,
encoder/embedding/reranking models, translation, vision-language and OCR,
speech recognition and synthesis, diffusion image and video generation,
segmentation, time-series forecasting, and neural operators.

Each model-family reference implementation keeps its knowledge in family-owned
builder, runtime, and E2E descriptors. These implementations serve as concrete
blueprints for modification and customization rather than hiding model behavior
behind a single generic integration. The repository also includes agent
instructions and model-local validation contracts so contributors can extend
one family without editing a hand-written global registry.

## Installation and setup

The development container supplies TensorRT, CUDA, Python, CMake, Ninja, and
the declared Python execution profiles. Start with the repository already
cloned and a terminal open at the checkout root. The host must have a
working NVIDIA driver, Docker Engine with NVIDIA Container Toolkit support,
network access, and enough disk space to build the image and model bundle. No
other host directory is assumed.

Build the development image for one target SM. `TORCH_CUDA_ARCH_LIST` uses the
dotted compute-capability form; for example, build for SM103 with:

```bash
docker build \
  --build-arg TRTMC_TORCH_CUDA_ARCH_LIST=10.3 \
  --tag trtmc-dev-sm103 .
```

For SM110, use:

```bash
docker build \
  --build-arg TRTMC_TORCH_CUDA_ARCH_LIST=11.0 \
  --tag trtmc-dev-sm110 .
```

When a prebuilt development image is published, replace the corresponding
local `docker build` with `docker pull`. Select the tag matching the target SM,
then start the source-mounted container. This example continues with SM103.

The host source path is resolved from Git, and `/src` is a container-local
mount point created by Docker; neither path needs to be known in advance.

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

### Native build configurations

Inside the container, use the same SM number selected for the image. Install
the editable Python builder once, then choose one native build configuration
below. Change `TRTMC_SM=103` to `TRTMC_SM=110` when targeting SM110.

```bash
TRTMC_SM=103
python -m pip install --no-deps -e . -C py-only=true
```

#### Default: standard TensorRT backend and all model DSOs

**Use case:** first-time setup and general development across the supported
model inventory. This is the recommended configuration for the Qwen smoke test
below.

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
DSO, but their CUDA code contains only the selected SM.

#### Focused: one model DSO

**Use case:** iterating on one runtime family when rebuilding every model DSO
would add unnecessary compile time. The target name uses the runtime owner ID,
not the Hugging Face repository ID or an internal test profile. For example,
Qwen models use the `qwen` owner:

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

Runtime owner IDs are the directory names under `src/runtime/models/`; each
directory's `MODEL.toml` declares its output DSO and runtime strategies.

#### Optional: TensorRT-RTX backend

**Use case:** building and running native bundles explicitly created with
`trtmc build --rtx`. Do not use this configuration for a standard TensorRT
bundle.

The generic development image includes the standard TensorRT SDK, not the
optional TensorRT-RTX SDK. Before configuring this variant, install the
matching `tensorrt_rtx` Python package, headers, and `libtensorrt_rtx.so` in
the development environment. Set the two SDK directories explicitly; no
installation path is assumed:

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

### Qwen smoke test

After the recommended default build, run the basic end-to-end smoke test:

```bash
TRTMC_BUILD_DIR="build-sm${TRTMC_SM}"
export TRTMC_MODEL_PLUGIN_DIR="$TRTMC_BUILD_DIR/models"

"$TRTMC_BUILD_DIR/trtmc" build Qwen/Qwen3-0.6B --output qwen3-0.6b.bundle
"$TRTMC_BUILD_DIR/trtmc" run qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

The first command downloads the checkpoint and builds its platform-specific
TensorRT engines. The smoke test succeeds when the second command completes
and generates `Paris`. See
[Getting Started](website/docs/getting-started/overview.md) for additional
prerequisites and troubleshooting.

## Supported models

See the [Supported Models documentation](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview/)
for the complete checkpoint matrix, including Hugging Face architecture,
TRTMC profile, precision, quantization, optimized-runtime dispatch, and the
dated GB300 performance snapshot.

<!-- Collaborative review anchor. -->
