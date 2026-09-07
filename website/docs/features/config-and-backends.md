---
title: Build and Runtime Options
---

TensorRT-Model-Connect keeps configuration at the boundary that owns it. There
is no central schema registry and no generic `--config` or `--set` layer.

## Build options

The Python build command accepts a small set of explicit cross-family inputs:

```bash
python -m tensorrt_model_connect build MODEL \
  --output model.bundle \
  --precision fp16 \
  --backend trt
```

The shared `BuildRequest` carries those values to exactly one
`families/<family>/model.py`. That plain `build(request, writer)` function must
either implement a value or reject it. It never inherits from a shared builder
and the core never substitutes another family.

Current shared fields cover the output path, task, precision, backend, sequence
or media bounds, batch size, tensor/context parallel size, quantization,
selected FP32 layers, dynamic KV cache, verbosity, and the optional direct graph
transform used by BYOK. A family may impose stricter rules.

If a model needs another option, keep it in the owning family unless more than
one independent family proves a stable model-independent contract.

## Backend identity

The bundle header names one backend:

- `trt` loads `libtrtmc_backend_trt.so`;
- `trt_rtx` loads `libtrtmc_backend_trt_rtx.so`.

The runtime also loads `libtrtmc_model_<family>.so`. Both files must be in the
explicit runtime root:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello"
```

The loader does not search environment variables, the current directory,
aliases, or secondary backend directories. It verifies that the backend and
Task identities returned by the DSOs match the bundle.

## Runtime options

Only three shared load options exist today:

| Option | Owner |
| --- | --- |
| `--kv-cache-size BYTES|GB|GiB` | Passed to a family that built a runtime-sized KV cache. |
| `--runtime-cache PATH` | Passed to the TensorRT-RTX backend. |
| `--cuda-graphs` | Passed to the TensorRT-RTX backend. |

`--runtime-cache` and `--cuda-graphs` are rejected for a standard TensorRT
bundle. Request controls such as sampling, image size, or transcription options
belong to the selected abstract Task API and are interpreted by its family
implementation.

## TensorRT-RTX

Build the family for the optional backend only when the TensorRT-RTX Python
package and native SDK are installed:

```bash
python -m tensorrt_model_connect build MODEL \
  --output model-rtx.bundle \
  --backend trt_rtx

trtmc run model-rtx.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache \
  --cuda-graphs \
  --prompt "Hello"
```

Backend selection does not change family ownership. The family still builds
the graph and implements the Task interface; the backend only implements the
abstract Engine API.
