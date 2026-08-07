---
title: Configure Runtime Behavior
description: Put a setting at build, bundle, load, or request time without crossing ownership boundaries.
---

First identify when the setting takes effect:

| Lifecycle | Examples | Rebuild required? |
| --- | --- | --- |
| Build | Precision, quantization, engine shapes, topology | Yes |
| Bundle default | Packaged model/runtime defaults | Yes, unless edited by supported tooling |
| Load | Backend/model DSO search, runtime cache, CUDA-graph policy, registered config | No |
| Request | Prompt, sampling, diffusion steps, language, input media | No |

Registered configuration uses a JSON file or repeatable overrides:

```bash
trtmc run model.trtfb \
  --config runtime.json \
  --set runtime.disable_cuda_graph=true \
  --prompt "Hello"
```

The native CLI validates explicit fields against registered schemas. Unknown
namespaces, fields, invalid types, and out-of-range values fail. An optimized
implementation can accept or reject the same public `LoadOptions` according to
its exact provider contract; it does not automatically inherit native config
semantics.

Use [Configuration Reference](../features/config-and-backends.md) for the live
schema catalog and backend/cache behavior. Use
[Quantization](../features/quantization.md) and
[Multi-Device Execution](../features/multi-device.md) for their model-owned
build contracts.
