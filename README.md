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
[Model Support](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/model-support) |
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
See [Model Support](website/docs/getting-started/model-support.md) for evidence
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

## Performance snapshot by model and platform

This table is a dated performance snapshot from the completed July 29, 2026
release comparison on NVIDIA GB300 at source revision
`508613d0bcc7003b123cf5be3d1b3f6e6c6cb667`. It covers 105 unique
single-process release model profiles across 76 families. Distributed profiles
and shorter L0 smoke profiles are outside this matrix. This table alone makes
no verified-support claim for anything absent from it.

The model ID and build-configuration columns come from each profile's E2E
manifest at that same source revision; `hf_id` is the CLI input and the TRTMC
profile is the internal test name.

The platform-specialization column is checkpoint-level integration metadata,
not a benchmark result. At this revision, TensorRT Edge-LLM is the only
integrated provider; its currently qualified Model Connect dispatch tuple is
dense Qwen on Linux x86_64 with NVIDIA A100 80GB PCIe (SM80) and FP16. A
`Qualified TRTMC dispatch target: Coming soon` entry means that the exact
`hf_id` appears in both the Edge-LLM supported-checkpoint list and this table,
but Model Connect does not yet claim a qualified dispatch target for that
checkpoint. For a checkpoint that already has a qualified tuple, `Additional
tuples: Coming soon` marks broader planned coverage. Provider selection still
requires an exact model ID, immutable revision, target, precision, and option
match; a provider shown on a row does not mean that the row's measured
configuration used that provider. `—` means that no platform-specialization
integration is currently identified for that exact checkpoint. See
[Qualified optimized implementations](website/docs/features/model-families.md#qualified-optimized-implementations)
for the current qualification boundary.

> **Platform-specialization rollout:** Platform specializations will be rolled
> out in phases, aligned with model coverage available in TensorRT Edge-LLM and
> TensorRT-Model-Connect releases. Each batch will document the exact qualified
> model, target, precision, and configuration tuples it adds.

Each row's evidence applies only to the exact `hf_id`, checkpoint resolution,
and build configuration exercised at that snapshot; an `hf_revision` is shown
when the manifest pins one. Untested same-family fine-tunes are best-effort
compatible, not verified supported checkpoints.

Each light compares TRTMC inference p50 with the row's declared reference under
matching workload, output, and timing contracts:

- **🟢 Green:** TRTMC is more than 5% faster than the reference.
- **🟡 Yellow:** TRTMC is within 5% of the reference.
- **🔴 Red:** TRTMC is more than 5% slower than the reference.
- **— Not supported:** the model profile is explicitly unsupported on that
  platform.

Bundle preparation, model loading, compilation when used, and warmup are
excluded from these infer-p50 values. Baselines are declared per row and are
not uniformly `torch.compile`: this snapshot contains 59 `torch.compile`, 34
Hugging Face eager, and 12 PyTorch eager references. The
[release comparison contract at the measured revision](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/508613d0bcc7003b123cf5be3d1b3f6e6c6cb667/benchmarks/performance/release.yaml)
identifies the reference and measurement policy for every row; its
[revision-matched documentation](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/508613d0bcc7003b123cf5be3d1b3f6e6c6cb667/benchmarks/performance/README.md)
explains how to reproduce and interpret the matrix.

A traffic light is emitted only after the row's output and measurement
contracts match. It is performance evidence for this exact snapshot, not a
general correctness or compatibility guarantee.

| Hugging Face model ID (`hf_id`, CLI input) | TRTMC profile | Build precision | Quantization | Platform specialization runtime provider | GB300 |
| --- | --- | --- | --- | --- | --- |
| `albert/albert-base-v2` | `albert-base` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/all-MiniLM-L6-v2` | `all-minilm-l6-v2` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/all-mpnet-base-v2` | `all-mpnet-base-v2` | `FP16` | None | — | 🟢 Green |
| `suno/bark` | `bark-large` | `FP32` | None | — | 🟢 Green |
| `suno/bark-small` | `bark-small` | `FP16` | None | — | 🟢 Green |
| `facebook/bart-base` | `bart-base` | `FP16` | None | — | 🟢 Green |
| `google-bert/bert-base-uncased` | `bert-base-uncased` | `FP16` | None | — | 🟢 Green |
| `BAAI/bge-small-en-v1.5` | `bge-small-en-v1.5` | `FP16` | None | — | 🟢 Green |
| `bigscience/bloom-560m` | `bloom-560m` | `FP32` | None | — | 🟢 Green |
| `almanach/camembert-base` | `camembert-base` | `FP16` | None | — | 🟢 Green |
| `nvidia/canary-1b-v2` | `canary-1b-v2` | `FP16` | None | — | 🟢 Green |
| `amazon/chronos-bolt-tiny` | `chronos-bolt-tiny-official` | `FP32` | None | — | 🟢 Green |
| `Salesforce/codegen-350M-mono` | `codegen-350m` | `FP16` | None | — | 🟢 Green |
| `YituTech/conv-bert-base` | `convbert-base` | `FP16` | None | — | 🟢 Green |
| `microsoft/deberta-base` | `deberta-base` | `FP16` | None | — | 🟢 Green |
| `deepseek-ai/DeepSeek-OCR-2` | `deepseek-ocr` | `FP16`<br />FP32 layers: `6, 7, 8, 9, 10, 11, 12` | None | — | 🟢 Green |
| `deepseek-ai/DeepSeek-V2-Lite` | `deepseek-v2-lite` | `FP16` | None | — | 🟢 Green |
| `yujiepan/deepseek-v3-tiny-random` | `deepseek-v2-tiny` | `FP16` | None | — | 🟢 Green |
| `distilbert/distilbert-base-uncased` | `distilbert-base-uncased` | `FP16` | None | — | 🟢 Green |
| `distilbert/distilgpt2` | `distilgpt2` | `FP16` | None | — | 🟢 Green |
| `facebook/dpr-ctx_encoder-single-nq-base` | `dpr-ctx-encoder` | `FP16` | None | — | 🟢 Green |
| `google/electra-base-discriminator` | `electra-base-discriminator` | `FP16` | None | — | 🟢 Green |
| `tiiuae/falcon-rw-1b` | `falcon-rw-1b` | `FP16` | None | — | 🟢 Green |
| `tiiuae/Falcon3-1B-Base` | `falcon3-1b` | `FP16` | None | — | 🟢 Green |
| `black-forest-labs/FLUX.2-dev` | `flux-2-dev` | `FP16` | None | — | 🟢 Green |
| `black-forest-labs/FLUX.2-dev` | `flux-2-dev-fp8` | `FP16` | `fp8_scales=data/flux2-fp8-scales.json` | — | 🟢 Green |
| `black-forest-labs/FLUX.1-schnell` | `flux-schnell` | `FP16` | None | — | 🟢 Green |
| `google/fnet-base` | `fnet-base` | `FP16` | None | — | 🟢 Green |
| `google/gemma-2-2b-it` | `gemma-2-2b` | `FP16` | None | — | 🟢 Green |
| `THUDM/glm-4-9b-hf` | `glm-4-9b` | `FP16` | None | — | 🟢 Green |
| `EleutherAI/gpt-neo-125m` | `gpt-neo-125m` | `FP16` | None | — | 🟢 Green |
| `openai/gpt-oss-20b` | `gpt-oss-20b` | `FP16` | None | — | 🟢 Green |
| `openai-community/gpt2` | `gpt2-125m` | `FP32` | None | — | 🟢 Green |
| `ibm-granite/granite-3.1-2b-base` | `granite-3.1-2b` | `FP16` | None | — | 🟢 Green |
| `internlm/internlm2-math-plus-1_8b` | `internlm2-1.8b` | `FP16` | None | — | 🟢 Green |
| `OpenGVLab/InternVL3-2B-hf` | `internvl3-2b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `OpenGVLab/InternVL3-8B-hf` | `internvl3-8b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `bytedance-research/Lance` | `lance-3b-x2t-image` | `BF16` | None | — | 🟢 Green |
| `nvidia/LocateAnything-3B` | `locateanything-3b` | `FP16` | None | — | 🟢 Green |
| `nvidia/magpie_tts_multilingual_357m`<br />Revision: `34d7e40da85cabc97f92198889b65cea27bc7fd1` | `magpie-tts-357m` | `FP32` | None | — | 🟢 Green |
| `state-spaces/mamba-130m-hf` | `mamba-130m` | `FP32` | None | — | 🟢 Green |
| `Helsinki-NLP/opus-mt-en-ru` | `marian-en-ru` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Minitron-4B-Depth-Base` | `minitron-4b-depth` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Minitron-4B-Width-Base` | `minitron-4b-width` | `FP32` (manifest default) | None | — | 🟢 Green |
| `mistralai/Mistral-7B-Instruct-v0.1` | `mistral-7b` | `FP16` | None | — | 🟢 Green |
| `ggml-org/stories15M_MOE` | `mixtral-stories-15m` | `FP16`<br />FP32 layers: `4` | None | — | 🟢 Green |
| `answerdotai/ModernBERT-base` | `modernbert-base` | `FP32` | None | — | 🟢 Green |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | `nemotron-3.5-asr-streaming-0.6b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | `nemotron-embed-vl-1b-v2` | `FP16` | None | — | 🟢 Green |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `nemotron-h-nano-9b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `nvidia/Nemotron-4-Mini-Hindi-4B-Base` | `nemotron-hindi-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/Nemotron-Labs-Diffusion-8B` | `nemotron-labs-diffusion-8b` | `BF16` | None | — | 🟢 Green |
| `nvidia/Nemotron-Mini-4B-Instruct` | `nemotron-mini-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | `nemotron-nano-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/llama-nemotron-rerank-vl-1b-v2` | `nemotron-rerank-vl-1b-v2` | `FP16` | None | — | 🟢 Green |
| `nvidia/nemotron-speech-streaming-en-0.6b` | `nemotron-speech-streaming-en-0.6b` | `FP16` | None | — | 🟢 Green |
| `facebook/nllb-200-distilled-600M` | `nllb-200-distilled-600m` | `FP16` | None | — | 🟢 Green |
| `allenai/OLMo-1B-hf` | `olmo-1b` | `FP16` | None | — | 🟢 Green |
| `allenai/OLMo-2-0425-1B` | `olmo2-1b` | `FP16` | None | — | 🟢 Green |
| `facebook/opt-125m` | `opt-125m` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `paraphrase-multilingual-minilm-l12-v2` | `FP16` | None | — | 🟢 Green |
| `ibm-granite/granite-timeseries-patchtsmixer` | `patchtsmixer-granite-official` | `FP16` | None | — | 🟢 Green |
| `ibm-research/patchtst-etth1-regression-distribution` | `patchtst-etth1-regression-distribution` | `FP16`<br />FP32 layers: `5` | None | — | 🟢 Green |
| `ibm-granite/granite-timeseries-patchtst` | `patchtst-granite-official` | `FP16` | None | — | 🟢 Green |
| `nvidia/personaplex-7b-v1` | `personaplex-7b` | `FP16`<br />FP32 layers: `0, 1` | None | — | 🟢 Green |
| `microsoft/Phi-tiny-MoE-instruct` | `phi-moe` | `FP16` | None | — | 🟢 Green |
| `microsoft/Phi-3-mini-4k-instruct` | `phi3-mini` | `FP16` | None | — | 🟢 Green |
| `microsoft/Phi-4-multimodal-instruct` | `phi4-multimodal` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | `pixart-sigma-1024` | `FP16`<br />FP32 layers: `0` | None | — | 🟢 Green |
| `EleutherAI/pythia-70m` | `pythia-70m` | `FP32` | None | — | 🟢 Green |
| `Qwen/Qwen-Image` | `qwen-image` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen-Image-2512` | `qwen-image-2512` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen-Image-Edit-2511` | `qwen-image-edit-2511` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `qwen25vl-3b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-fp16` | `FP16` | None | TensorRT Edge-LLM<br />Qualified dispatch tuple: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional tuples: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-fp8` | `BF16` | `format=fp8`<br />`scale_source=modelopt`<br />`calibration_samples=64` | TensorRT Edge-LLM<br />Qualified dispatch tuple: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional tuples: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-topp` | `FP16` | None | TensorRT Edge-LLM<br />Qualified dispatch tuple: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional tuples: Coming soon | 🟢 Green |
| `Qwen/Qwen3-4B-Instruct-2507` | `qwen3-4b-instruct-2507` | `FP16` | None | TensorRT Edge-LLM<br />Qualified dispatch tuple: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional tuples: Coming soon | 🟢 Green |
| `Qwen/Qwen3-30B-A3B` | `qwen3-moe-30b-a3b` | `FP16` | None | — | 🟢 Green |
| `amd-quark/tiny-random-qwen3_moe` | `qwen3-moe-tiny-random` | `FP16` | None | — | 🟢 Green |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `qwen3-omni-30b-a3b-instruct` | `BF16` | None | — | 🔴 Red |
| `Qwen/Qwen3-VL-2B-Instruct` | `qwen3-vl-2b` | `FP16`<br />FP32 layers: `0, 1, 2` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `Qwen/Qwen3.5-9B` | `qwen35-9b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `nvidia/Riva-Translate-4B-Instruct-v1.1` | `riva-translate-4b` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/roberta-base` | `roberta-base` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/roberta-large` | `roberta-large` | `FP16` | None | — | 🟢 Green |
| `RWKV/rwkv-4-169m-pile` | `rwkv-169m` | `FP32` | None | — | 🟢 Green |
| `facebook/sam-vit-base` | `sam-vit-base` | `FP16` | None | — | 🟢 Green |
| `facebook/sam3` | `sam3` | `FP32` | None | — | 🟢 Green |
| `Efficient-Large-Model/SANA-WM_bidirectional` | `sana-wm-bidirectional` | `BF16` | None | — | 🟡 Yellow |
| `nvidia/segformer-b0-finetuned-ade-512-512` | `segformer-b0-ade` | `FP16` | None | — | 🟢 Green |
| `stabilityai/stablelm-2-1_6b` | `stablelm2-1.6b` | `FP16` | None | — | 🟢 Green |
| `bigcode/starcoder2-3b` | `starcoder2-3b` | `FP16` | None | — | 🟢 Green |
| `google-t5/t5-small` | `t5-small` | `FP16` | None | — | 🟢 Green |
| `google/timesfm-2.0-500m-pytorch` | `timesfm-2.0-500m-official` | `FP32` | None | — | 🟢 Green |
| `timm/vit_base_patch16_224.augreg_in21k_ft_in1k` | `timm-vit-base-p16-224-augreg-in21k-ft-in1k` | `FP16` | None | — | 🟢 Green |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `tinyllama-1.1b` | `FP16` | None | — | 🟢 Green |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `wan21-t2v-1.3b` | `FP16`<br />FP32 layers: `24` | None | — | 🟢 Green |
| `Wan-AI/Wan2.2-TI2V-5B`<br />Revision: `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | `wan22-ti2v-5b` | `BF16` | None | — | 🟢 Green |
| `openai/whisper-large-v3-turbo` | `whisper-large-v3-turbo` | `FP32` | None | — | 🟢 Green |
| `openai/whisper-tiny` | `whisper-tiny-fp16` | `FP16`<br />FP32 layers: `0` | None | — | 🟢 Green |
| `facebook/xglm-564M` | `xglm-564m` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/xlm-roberta-base` | `xlm-roberta-base` | `FP16` | None | — | 🟢 Green |
| `xlnet/xlnet-base-cased` | `xlnet-base` | `FP16` | None | — | 🟢 Green |
| `Tongyi-MAI/Z-Image-Turbo` | `z-image-turbo` | `FP16`<br />FP32 layers: `2, 3, 4, 7, 8` | None | — | 🔴 Red |

<!-- Collaborative review anchor. -->
