---
title: Configure Runtime Behavior
description: Put a setting at build, load, or request time without crossing ownership boundaries.
---

First identify when the setting takes effect:

| Lifecycle | Examples | Rebuild? |
| --- | --- | --- |
| Build | Precision, engine bounds, topology, quantization, dynamic-KV graph | Yes |
| Family bundle state | Tokenizer, preprocessing weights, engine layout, private runtime JSON | Yes |
| Load | Runtime root, runtime-sized KV capacity, TensorRT-RTX cache/CUDA graphs, BYOK binding | No |
| Request | Prompt, sampling, language, media, denoising steps, seed | No |

The current API has no generic `--config` / `--set` registry. Build values are
typed `BuildRequest` fields; runtime values are loader inputs or typed Task
configs. Family-only state remains in family sections and implementation code.

```bash
trtmc run model.bundle \
  --kv-cache-size 4GiB \
  --prompt "Hello" \
  --temperature 0
```

Only compatible family bundles accept runtime-sized KV cache. Likewise,
`--runtime-cache` and `--cuda-graphs` are valid only for a `trt_rtx` bundle.
Unsupported values fail; they are not silently ignored.

See [Configuration and Backends](../features/config-and-backends.md),
[Quantization](../features/quantization.md), and
[Multi-Device Execution](../features/multi-device.md).
