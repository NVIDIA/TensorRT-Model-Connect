---
title: Quantization and Runtime Knobs
---

# Quantization and runtime knobs

TRTMC separates values that shape a bundle from values used while loading or
executing it. There is no global configuration registry after the family
isolation cutover.

## Build-time choices

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  -o qwen.bundle \
  --precision bf16 \
  --max-sequence-length 4096 \
  --max-batch-size 1 \
  --dynamic-kv-cache
```

The public builder also exposes `--backend trt|trt_rtx`, tensor/context
parallel sizes, image/video dimensions, quantization, and repeatable
`--fp32-layer` overrides. Each family validates the subset it supports.

## Quantization

Pass an explicit family-supported mode:

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  -o qwen-fp8.bundle \
  --precision bf16 \
  --quantization fp8
```

Quantization routing, scale acquisition, layer exceptions, graph construction,
and validation are family-owned. Some families reject every quantization mode;
others package calibrated or precomputed scales. Check the selected
`families/<family>/model.py`, `support.py`, and tests before choosing a value.
Do not copy a mode merely because another family accepts it.

`--fp32-layer INDEX` may be repeated only when the family implements that
override. It is a build input, not a request-time tuning option.

## Runtime loading choices

Every execution command requires the installed runtime directory:

```bash
trtmc run qwen.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache /tmp/trtmc-cache \
  --cuda-graphs \
  --kv-cache-size 4096 \
  --prompt "Hello" \
  --max-new-tokens 64
```

Use only options shown by the installed CLI. There is no `--backend-dir`,
model-plugin search path, or provider selection flag. `--backend trt_rtx` is a
build-time request for a family that supports complete-network TRT-RTX offload.

## Request choices

Prompt, media paths, sampling values, diffusion steps, transcription options,
and output paths affect one typed task call. They do not rewrite an existing
bundle or switch its family implementation.

For comparisons, change one boundary at a time and retain the exact build,
load, and request arguments with correctness evidence.
