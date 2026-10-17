---
title: Model Families
---

A model family is one complete vertical slice under `families/<family>/`:

```text
families/<family>/
  support.py
  model.py
  requirements.txt          # optional
  runtime/CMakeLists.txt
  runtime/*.cpp
  tests/test_e2e.py
  tests/manifests/*.json
  tests/thresholds/*.json    # optional
```

## Discovery and build

`support.py` performs dependency-free exact matching of root model metadata and
declares the supported tasks plus default. It must not import the family build,
TensorRT, PyTorch, Transformers, or another family.

After exactly one match, the core imports only that family's `model.py` and
calls its plain `build(request, writer)` function. The family owns config and
weight interpretation, TensorRT graph topology, precision/quantization policy,
engine construction, and bundle-section semantics.

### K2-Horizon-Uno task contract

`IFM/K2-Horizon-7B-Uno` is an adapter-only checkpoint. The independent
`k2_horizon_uno` family resolves its pinned `IFM/K2-Horizon-7B` base and
compiles the rank-128 conditional LoRA into one BF16 TensorRT engine. Build it
directly from the adapter ID:

```bash
python -m tensorrt_model_connect build IFM/K2-Horizon-7B-Uno \
  --revision ec92bbd768f4a404319625204544782e3377bcd7 \
  --precision bf16 \
  -o k2-horizon-7b-uno.bundle
```

Run the qualified high-reasoning chat path with the family-owned linear mode:

```bash
trtmc run k2-horizon-7b-uno.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Reply with the word OK." \
  --max-new-tokens 26 \
  --temperature 0 \
  --top-k 1 \
  --generation-mode linear_spec_lora \
  --block-length 8 \
  --use-chat-template true \
  --enable-thinking true
```

The initial runtime supports batch-one deterministic greedy generation with
linear Psi-Spec and block lengths from 1 through 8. Block 8 is the qualified
default. It also supports the pinned single-user, high-reasoning chat template
and an autoregressive control mode. String prompts are currently limited to
ASCII and fail closed before tokenization otherwise. Stochastic sampling, tree
verification, batching, tensor parallelism, quantization, the 0.9B adapter, and
long-context qualification remain outside this contract. The
committed-token-per-forward receipt is an algorithmic diagnostic, not a
wall-clock speedup claim.

### Gemma4 Edge ONNX execution

Standalone Gemma4 text/media offload also maps the published 26B-A4B NVFP4
checkpoint on native SM12x, preserving its FP8 KV cache without calibration.
Its build and text/image commands execute, but recorded MC/direct text/image
divergence and an unavailable independent FP32 reference prevent an accuracy
qualification claim. See the [owning Gemma adapter notes](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md)
for exact boundaries; quality failures do not disable executable requests.

Use `trtmc gemma build MODEL -o model.bundle` with the owning
family's options. `trtmc gemma build --help` works offline without
a checkpoint or GPU imports. This uses the existing
[family CLI protocol](../extend/family-cli.md), not an extension to the shared parser.

The Gemma family also owns explicit Gemma4-12B target/assistant MTP and
Gemma4-12B/DSpark block7 execution through the optional pinned native Edge-LLM
SDK. These are text-only FP16 paired profiles, qualified on SM80; selecting a
Gemma4 checkpoint alone does not enable them or claim standalone native support.

Provision the [native SDK](../user-guides/configure-runtime.md#optional-native-edge-llm-sdk),
then pass `--execution-variant mtp` or `--execution-variant dspark` with
`--companion draft=/path/to/checkpoint` to the build CLI. MTP is greedy-only;
DSpark preserves the supported sampling controls. Exact checkpoint revisions,
capacity bounds, validation results, unsupported controls and source-faithful
chat-template handling are documented in the
[owning Gemma recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md).
These local qualifications are separate from the registered manifest inventory
and do not imply that CI executes the paired cases.

### Qwen3.6 / Qwen3.8 Edge-LLM execution

The family owns ordinary original-weight FP16 TP2, ordinary mixed-NVFP4 TP1,
and explicit DSpark/DFlash2 ONNX execution through the optional Edge-LLM 0.11.0
SDK. Use `trtmc qwen3_8 build MODEL -o model.bundle`; offline help and options
use the existing [family CLI protocol](../extend/family-cli.md), not shared
parser extensions. For a pair, add `--execution-variant dspark` or `dflash` and
`--companion draft=/path/to/draft`.

The adapter additionally maps uniform NVFP4/INT4-AWQ embedded MTP and INT4-AWQ
DFlash2. These have Python/native contract-test coverage, not local model-engine
or accuracy qualification. For MTP, use `--execution-variant mtp` without an
external companion. Prepared checkpoint metadata must match the owning recipe;
Model Connect does not calibrate or quantize weights during this integration.

Local text requests include exact independent FP32 matches, with separate
quality failures retained. Qwen3.8 original single-image passed; ordered-image
and video accuracy failed. Mixed-NVFP4
ordinary and both paired raw-text profiles build and run, matching direct Edge
exactly, but fail the independent FP32 NED gate. Accuracy failures remain
executable and are not described as fully qualified. Original-weight speculative
execution and batch-8 MTP remain outside this adapter's mapped scope.

Provision the [native SDK](../user-guides/configure-runtime.md#optional-native-edge-llm-sdk).
See the [owning recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/qwen3_8/edge_llm/README.md)
for exact checkpoints, commands, platform limits, and evidence scope. Local
profile results do not imply registered CI or catalog-wide qualification.


### Nemotron-H Edge-LLM execution

Use `trtmc nemotron_h build MODEL -o model.bundle` with the owning
family's options. `trtmc nemotron_h build --help` works offline without
a checkpoint or GPU imports. This uses the existing
[family CLI protocol](../extend/family-cli.md), not an extension to the shared parser.

The Nemotron-H family owns optional pinned Edge-LLM whole-network offload.
Ordinary compatible text builds use the experimental builder; the explicit
Lightning NVFP4/DFlash pair uses ONNX with
`--execution-variant dflash --companion draft=/path/to/draft`.
Native fallback is attempted only during ordinary preparation; it cannot
interpret unsupported packed checkpoints or replace a requested pair.

See the [owning Nemotron-H recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_h/edge_llm/README.md)
for the six recorded ordinary profiles, the qualified greedy DFlash pair,
immutable revisions and unchanged quality gates. These are bounded historical
local qualifications, not catalog-wide or current-head CI proof. The separate
direct-Edge 9B-NVFP4 quality failure and longer-context failures remain open.
The original plain 9B case is registered in the owning E2E inventory; the other
exact profiles are not implied to run in CI.

## Runtime and validation

The directory name is also the runtime DSO identity:
`libtrtmc_model_<family>.so`. The family factory implements one or more
abstract interfaces in `trtmc/task.h` and owns all preprocessing,
postprocessing, dispatch, state, sampling, and engine binding.

Tests, exact checkpoint manifests, fixtures, thresholds, and reference/oracle
logic live in the same directory. A family with extra build or reference
dependencies declares them in its own plain `requirements.txt`.

## Live inventory

The website's [Models & Recipes](../models-recipes/overview.md) pages are
generated from `families/*/support.py` and `tests/manifests/*.json`. Validate the
physical inventory with:

```bash
python3 -m tools.model_ci validate
```

The project covers text decoders and encoders, MoE and recurrent networks,
vision-language, speech/audio, diffusion, perception, time-series, robotics,
and world-model tasks. Support remains exact and model-owned: a similar
architecture, parser option, or sibling family's passing test is not evidence
for another checkpoint.

## Isolation rule

Families never import, include, or link siblings. Similar model code is copied
by design so a team can implement, validate, change, and revert one family
without modifying another. Shared code is limited to model-agnostic contracts
and mechanics described in the
[Architecture](../architecture/ai-native-horizontal-scaling.md).

### Optional InternVL Edge execution

Use `trtmc internvl build MODEL -o model.bundle` with the owning
family's options. `trtmc internvl build --help` works offline without
a checkpoint or GPU imports. This uses the existing
[family CLI protocol](../extend/family-cli.md), not an extension to the shared parser.

The InternVL family can offload the recorded original-source InternVL3
1B/2B/8B HF configurations on native x86 SM80 and 14B on native x86 SM120 to the
pinned Edge-LLM 0.10.1 SDK. These FP16, batch-one, TP-one profiles have historical
local public/direct/HF qualification; this is not catalog-wide or fresh-head CI
qualification. The existing 2B/8B image-health test reads actual Edge visual
features through a small family-owned test helper without changing its gates.
Quantized sources, InternVL3.5, and multi-image public requests are excluded.
See the [family recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/internvl/edge_llm/README.md)
for exact source revisions, evidence boundaries, and replay requirements.

### Optional Qwen3.5 Edge execution

Use `trtmc qwen3_5 build MODEL -o model.bundle` with the owning
family's options. `trtmc qwen3_5 build --help` works offline without
a checkpoint or GPU imports. This uses the existing
[family CLI protocol](../extend/family-cli.md), not an extension to the shared parser.

The Qwen3.5 family maps the recorded original-source dense 0.8B/2B/4B/9B
configurations (Instruct and Base), embedded MTP, and explicit 4B/9B DFlash
pairs to the pinned Edge-LLM 0.11.0 SDK on native x86 SM80/SM120, FP16 compute,
batch one. Dense 27B uses TP2 for its LLM and TP1 for its visual component.
The SM120 route preserves admitted ModelOpt weight formats; the recorded 2B
FP8-KV and NVFP4 LM-head component flows are checkpoint-driven.
Compatible PyPI Python tools are discovered automatically, with the source-built
SDK interpreter as fallback; a matching native C++ SDK is still required.
Execution admission is not a blanket accuracy claim: recorded quality-only
failures remain runnable and existing gates are unchanged. Qwen3 or older,
MoE, Qwen3.6/3.8 and speculative TP2 are outside this owner.
See the [family recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/qwen3_5/edge_llm/README.md)
for commands, paired execution, validation boundaries and known quality issues.

Visual bundles expose the existing typed video and mixed image/video APIs;
decoded frame order and optional timestamps are preserved. The recorded 2B
two-frame appearance case passed Model Connect/direct parity and independent
FP32 NED0.0; it is not a motion-quality or all-configuration qualification.
The generic image CLI is unchanged. See the owning recipe for API boundaries.

## Optional Llama Edge execution

Use `trtmc llama build MODEL -o model.bundle` with the owning
family's options. `trtmc llama build --help` works offline without
a checkpoint or GPU imports. This uses the existing
[family CLI protocol](../extend/family-cli.md), not an extension to the shared parser.

The Llama family owns an optional native Edge-LLM 0.10.1 route for original
unquantized sources with FP16 compute. Recorded ordinary Llama 3.1 8B and
3.2 1B/3B profiles use SM80; the explicit Llama 3.1 8B + EAGLE3 pair uses SM120.
Other requests retain native behavior, and a failed explicit pair never silently
becomes base-only decoding. Quantized Llama routes are outside this publication.

These are bounded historical build/inference results, not catalog-wide support
or an assertion that every profile is registered in CI. Passing engine payloads
were retired; publication checks are reported separately. See the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/llama/edge_llm/README.md)
for exact revisions, capacities, controls and validation boundaries.
