---
title: Quick Start
---

This quick start builds one text-generation bundle, inspects it, and runs it through the C++ runtime.

Complete [Installation](installation.md) first. If you installed a release
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
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-0.6b.trtfb \
  --precision fp16 \
  --max-cache-length 256
```

`trtmc build` resolves the HuggingFace model, selects the matching Python family plugin, builds TensorRT engine plan bytes, and writes a self-contained `.trtfb` bundle.

First builds can be slow because the builder may download model files and compile TensorRT engines. If the command fails before TensorRT starts, check model ID, HuggingFace auth, network/cache, and Python dependencies first.

## 3. Inspect The Bundle

Inspect the bundle:

```bash
$TRTMC inspect /tmp/qwen3-0.6b.trtfb
$TRTMC inspect /tmp/qwen3-0.6b.trtfb --list-engines
```

Expected fields include:

```text
Model ID:           Qwen/Qwen3-0.6B
Family:             qwen
Runtime strategy:   decoder_kv_cache
Precision:          fp16
```

Inspection should become the first debugging habit. The important fields are `family`, `precision`, `runtime_strategy`, engine sections, tokenizer assets, and TensorRT compatibility metadata.

## 4. Run Deterministic Inference

```bash
$TRTMC run /tmp/qwen3-0.6b.trtfb \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy
```

`--greedy` makes the smoke test deterministic: each step chooses the highest-score token instead of sampling randomly. For Qwen3-0.6B, the runtime should log `Using native BPE tokenizer`; no `--hf-python` path is needed for this text-generation smoke test.

Add `--hf-python /opt/venv/bin/python` only when a runtime strategy still needs helper Python code, such as speech-to-speech prompt handling or a legacy fallback path.

## 5. Interpret The Result

If generation succeeds, you have proven this path:

```mermaid
flowchart LR
  Build["trtmc build"] --> Bundle["/tmp/qwen3-0.6b.trtfb"]
  Bundle --> Inspect["inspect metadata"]
  Bundle --> Load["trtmc::load"]
  Load --> Strategy["decoder_kv_cache"]
  Strategy --> Generate["IPipeline::generate"]
  Generate --> Text["TextResult"]
```

If generation fails, classify the failure before changing code:

| Failure | Usually means |
| --- | --- |
| Build cannot download model | HuggingFace model ID, auth, network, or cache problem. |
| Build fails inside TensorRT | Unsupported graph, shape/profile issue, or TensorRT environment issue. |
| Inspection fails | Bundle was not written correctly, the path is wrong, or the runtime library environment is incomplete. |
| Runtime says no plugin registered | The binary was built without the plugin for the bundle's `runtime_strategy`. |
| Output differs between runs | Sampling is enabled. Use `--greedy` or a fixed `--seed` for smoke tests. |

## Jetson Thor: Wan2.2 720p In Two Commands

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
  -o wan22-thor.trtfb

trtmc generate-video wan22-thor.trtfb \
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

For a board that does not already have Docker, Model Connect, or a checkpoint
cache, follow [Wan2.2 On A Fresh Jetson Thor](wan2-2-thor-from-scratch.md).
That customer image consumes only the official TensorRT 11.1.0.106 NGC image.

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

- [Build and Run](build-and-run.md) covers common tasks for text, vision-language, audio, diffusion, segmentation, and time-series bundles.
- [Model Support](model-support.md) explains the current supported model surface from the manifest set.
- [Inspect Bundles](../tutorials/beginner/inspect-bundles.md) teaches the artifact-debugging workflow.
- [CLI Reference](../api/cli-reference.md) lists the build and runtime command surfaces.
