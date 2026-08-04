---
title: Your First NLP Inference
---

import Diagram from '@site/src/components/Diagram';

This is the single first-inference path for the site. It builds one
text-generation bundle, inspects it, and runs it through the native C++
runtime.

Complete [Prerequisites and Environment](environment-and-repro.md), then
[Installation](installation.md), before starting. If you installed a release
wheel, the command is `trtmc`. If you built from source in the dev container,
the command is `./build/trtmc`.

```bash
TRTMC=trtmc
# Source build alternative:
# TRTMC=./build/trtmc
```

## 1. Prove The Tools Are Available

```bash
$TRTMC version
```

Expected signals:

```text
trtmc 0.1.0
TRT support: yes
```

If source-built `./build/trtmc` fails with a missing shared library, you are
probably outside the dev container or missing its runtime library paths.
If `trtmc` from a wheel fails, check that you installed the
`manylinux_2_39_aarch64` wheel for your Python version and that the host has
compatible NVIDIA driver/CUDA runtime libraries.

## 2. Build A Bundle

```bash
$TRTMC build Qwen/Qwen3-0.6B
```

`trtmc build` resolves the Hugging Face model and family. Eligible dense Qwen3
declares a model-owned native default, so this model-only command skips the
optimized-provider probe and builds BF16 split prefill/decode engines with the
checkpoint's full context capacity. The omitted output path is derived as
`Qwen3-0.6B.bundle`.

First builds can be slow because the builder may download model files and compile TensorRT engines. If the command fails before TensorRT starts, check model ID, Hugging Face auth, network/cache, and Python dependencies first.

## 3. Inspect The Bundle

Inspect the bundle:

```bash
$TRTMC inspect Qwen3-0.6B.bundle
$TRTMC inspect Qwen3-0.6B.bundle --list-engines
```

Expected fields include:

```text
Model ID:           Qwen/Qwen3-0.6B
Family:             qwen
Runtime strategy:   qwen_decoder_kv_cache
Precision:          bf16
```

Inspection should become the first debugging habit. For this native bundle,
the important fields are `family`, `precision`, `runtime_strategy`,
`max_cache_length` (40960 for this checkpoint), `engine_plan` (decode),
`prefill_engine_plan`, tokenizer assets, and TensorRT compatibility metadata.
For an
optimized bundle, the regular inspector can confirm that
`optimized_runtime.json` and the embedded artifact section names are present.
It does not currently decode the descriptor's implementation/profile identity,
and `--list-engines` does not treat embedded optimized-runtime `.engine` files
as native plan sections.

## 4. Run Deterministic Inference

```bash
$TRTMC run Qwen3-0.6B.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

`--greedy` makes token selection deterministic for a fixed bundle and runtime
environment: each step chooses the highest-score token instead of sampling
randomly. It does not guarantee identical output after changing the checkpoint,
engine build, TensorRT/CUDA cohort, hardware/numeric path, or runtime code. For
Qwen3-0.6B, the runtime should log `Using native BPE tokenizer`; no
`--hf-python` path is needed for this text-generation smoke test.

Add `--hf-python /opt/venv/bin/python` only when a runtime strategy still needs helper Python code, such as speech-to-speech prompt handling or a legacy fallback path.

## 5. Interpret The Result

If generation succeeds, you have proven this path:

<Diagram
  src="/img/diagrams/getting-started/qwen3-first-inference.svg"
  alt="Qwen3 first-inference path from trtmc build through bundle inspection and text generation"
  caption="A successful first run proves bundle construction, inspection, native strategy dispatch, and deterministic generation."
/>

If generation fails, classify the failure before changing code:

| Failure | Usually means |
| --- | --- |
| Build cannot download model | Hugging Face model ID, auth, network, or cache problem. |
| Build fails inside TensorRT | Unsupported graph, shape/profile issue, or TensorRT environment issue. |
| Inspection fails | Bundle was not written correctly, the path is wrong, or the runtime library environment is incomplete. |
| Runtime says no plugin registered | The strategy has no manifest owner, or its owning model DSO is missing/unloadable from the model-plugin search path. |
| Output differs between runs | First check sampling and use `--greedy` or a fixed `--seed`; if it persists, compare the exact bundle, checkpoint revision, runtime/software cohort, hardware path, and logs. |

## Optional Advanced Example: Jetson Thor Wan2.2 720p

This example is not required for the newcomer milestone above. Continue to the
Learning Path if your goal is only to complete the first NLP inference.

Use a Model Connect wheel built against the official TensorRT 11.1.0.106
release for CUDA 13.3. Its pinned public TensorRT dependency is installed
automatically; the `wan` extra is needed only while the bundle is built:

```bash
python3.12 -m venv ~/.venvs/trtmc
. ~/.venvs/trtmc/bin/activate
python -m pip install \
  './tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl[wan]'
```

Then run these two native Model Connect commands:

```bash
trtmc build Wan-AI/Wan2.2-TI2V-5B \
  --model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --fp8 \
  -o wan22-thor.bundle

trtmc generate-video wan22-thor.bundle \
  --set wan2_2_ti2v.easycache_enabled=true \
  --set wan2_2_ti2v.easycache_threshold=1.0 \
  --set wan2_2_ti2v.easycache_max_consecutive_reuse=4 \
  --set wan2_2_ti2v.late_cfg_enabled=true \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
  --output wan22-frames \
  --seed 42
```

The first command downloads the pinned checkpoint from Hugging Face, verifies
its contents, loads the packaged FP8 scales, and builds the target-specific
TensorRT bundle. No local checkpoint path, quantization JSON, plugin path, or
backend selector is needed.

The official bundle profile already supplies 1280x704, 121 frames, 50 steps,
CFG 5, flow shift 5, and 24 FPS. The runtime defaults supply the 7/2 exact-step
windows, and PNG output automatically selects up to eight writer threads.
Success ends with:

```text
Generated image: 1280x704 (121 frames)
```

The command writes `wan22-frames/frame_0000.png` through
`wan22-frames/frame_0120.png`. PyTorch is used only to read the checkpoint
during the build; generation runs through the native C++/TensorRT runtime.

The official TensorRT 11.1.0.106 path requires a fresh full 121-frame visual
and performance qualification on Jetson AGX Thor. Measurements and visual
receipts collected with earlier internal SDK builds do not qualify this
official-release path and must not be used as its latency or quality claim.

TensorRT plans must be built on the target Thor. Wan2.2 itself is not
Thor-only: the packaged FP8 profile also supports GB300 SM 10.3, while other
supported GPUs can omit `--fp8` and the four EasyCache `--set` arguments
to use the portable BF16 path.

## What To Read Next

- [Learning Path](../learning-path.md) continues from this bundle and orders
  the tutorials from beginner concepts through advanced validation and
  extension work.
- [Model Recipes](build-and-run.md) is an optional index for other modalities.
- [Model Support](model-support.md) explains the current supported model surface from the manifest set.
- [Inspect Bundles](../tutorials/beginner/inspect-bundles.md) teaches the artifact-debugging workflow.
- [CLI Reference](../api/cli-reference.md) lists the build and runtime command surfaces.

{/* Collaborative review anchor. */}
