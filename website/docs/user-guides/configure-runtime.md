---
title: Configure Runtime Behavior
description: Put each setting at build, load, or request time without crossing ownership boundaries.
---

First identify when a setting takes effect:

| Lifecycle | Examples | Rebuild required? |
| --- | --- | --- |
| Build | Precision, backend, engine shapes, topology, quantization | Yes |
| Bundle | Family-owned engine, metadata, tokenizer, and processor sections | Yes |
| Load | Runtime root, runtime-sized KV allocation, TensorRT-RTX cache, CUDA Graphs | No |
| Request | Prompt, sampling, input media, denoising steps, language | No |

## Explicit runtime root

Every execution command requires the one directory containing the compatible
runtime loader, backend DSO, and selected family DSO:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 32
```

The loader does not search environment variables, the current directory, or
another installation.

## Runtime-sized KV cache

A family built with `--dynamic-kv-cache` can accept an explicit allocation at
load time:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --kv-cache-size 1GiB \
  --prompt "Hello"
```

This option is meaningful only when the selected family built the matching
bundle contract.

## TensorRT-RTX controls

A bundle built with `--backend trt_rtx` can use its backend's load controls:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache \
  --cuda-graphs \
  --prompt "Hello"
```

Family code owns model behavior; the backend owns engine execution. Neither
surface changes another family's policy.

Use `trtmc help` for the current native command surface and
`python -m tensorrt_model_connect build --help` for build options.
