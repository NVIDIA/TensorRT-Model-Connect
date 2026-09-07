---
title: Build Pipeline
---

The Python build path has one resolver and one family call:

```text
model ID/local snapshot
  -> load root config.json or model_index.json metadata
  -> require exactly one families/*/support.py match
  -> choose family default task or validate --task
  -> import the selected families.<family>.model
  -> call build(BuildRequest, BundleWriter)
  -> atomically publish format-1 bundle
```

## Resolution

`core/builder/tensorrt_model_connect/model_support.py` loads standard root
identity metadata and dependency-free family support declarations. Matching is
exact after punctuation normalization. Zero matches means unsupported; more
than one means conflicting ownership. Repository names, scoring, priority, and
first-match fallback are not valid resolution mechanisms.

Only the selected `model.py` and its family dependencies are imported. This
keeps unrelated families isolated from dependency and import failures.

## Build request

The resolved `BuildRequest` carries the local model directory, output path,
family, non-empty task, backend, precision, shape bounds, batch and parallel
sizes, family-owned quantization selection, FP32 layer overrides, direct
dynamic-KV opt-in, and optional graph transform. Each family must implement or
explicitly reject every non-default request it receives.

## Family build

`families/<family>/model.py` exposes a plain `build(request, writer)` function.
It reads model config and weights, constructs the TensorRT network and engines,
and writes family-owned named sections. Builder inheritance and shared model
topology helpers are forbidden.

The graph-transform callback, when present, receives the live TensorRT network
immediately before serialization. This is the build-time half of the explicit
[TVM-FFI BYOK](../features/tvm-ffi.md) boundary.

## Failure and publication

Family exceptions are preserved; another family is never attempted. The
writer validates header fields and section names, streams large sections, and
publishes the output only after a successful build. A failed build removes its
temporary output and never presents a partial bundle as complete.
