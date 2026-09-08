---
title: Configuration and Backends
---

The current architecture uses typed, explicit inputs instead of a configuration
registry.

## Configuration ownership

| Layer | Public surface | Owner |
| --- | --- | --- |
| Build | `BuildRequest` and `python -m tensorrt_model_connect build` flags | Shared typing/validation; selected family implements or rejects the value. |
| Runtime task | Typed configs in `core/runtime/include/trtmc/task.h` | Public Task contract and selected family implementation. |
| Family state | Named bundle sections such as `runtime.json` | The owning family only. |
| Backend load | Runtime root, optional cache/CUDA graph settings | Exact loader and selected backend. |

There is no current `--config` / `--set` surface, schema registry, layered
defaults object, or effective-config artifact. Family-specific policy stays in
`families/<family>/`; a stable model-agnostic input is added to the narrow
shared request or Task type only when justified.

See [Configuration Boundaries](../extend/add-config-schema.md) for contributor
guidance.

## Native backend DSOs

The bundle header names one backend:

| Header value | Runtime DSO | Contract |
| --- | --- | --- |
| `trt` | `libtrtmc_backend_trt.so` | Standard TensorRT Engine implementation. |
| `trt_rtx` | `libtrtmc_backend_trt_rtx.so` | Optional TensorRT-RTX Engine implementation. |

Both are loaded from the same required runtime root as the family DSO:

```bash
trtmc run model.bundle \
  --prompt "Hello"
```

The loader does not accept a separate backend/model-plugin directory and does
not search environment variables or fallback paths. A standard backend bundle
rejects `--runtime-cache` and `--cuda-graphs`; those settings are forwarded
only for a `trt_rtx` bundle.

## Build-time backend selection

```bash
python -m tensorrt_model_connect build MODEL \
  --backend trt \
  --output model.bundle
```

Selecting `trt_rtx` changes the bundle backend identity and requires the
corresponding family path and installed TensorRT-RTX backend DSO. Parser
acceptance alone is not a support claim; validate the exact family, checkpoint,
backend, hardware, and Task behavior.
