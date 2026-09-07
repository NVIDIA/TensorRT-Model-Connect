---
title: Add an Optional Backend or Kernel
description: Use the existing backend or BYOK boundary instead of a second model-runtime framework.
---

The retired optimized-runtime provider/profile path has no equivalent in the
current architecture. Do not add profile registries, embedded implementation
trees, artifact digests, or fallback dispatch.

Choose one existing boundary.

## Engine-wide backend

Use a backend when multiple real families need the same TensorRT execution
implementation. The backend implements the abstract Engine API and remains
independent of model families.

TensorRT-RTX is the existing optional example:

```bash
python -m tensorrt_model_connect build MODEL \
  --backend trt_rtx \
  --output model.bundle

trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache kernels.cache \
  --cuda-graphs \
  --prompt "Hello"
```

The selected family still owns the graph and runtime pipeline.

## Model-specific TensorRT plugin

Compile a real model-specific custom plugin directly into the owning family
DSO. Do not move it into a shared backend merely because another model has
similar code.

## Bring your own kernel

Use the existing BYOK graph-transform and TVM-FFI bridge for an explicit custom
kernel. The build-time callback receives the live TensorRT network immediately
before serialization; runtime loads the explicitly named kernel DSO and
function.

See [Bring Your Own Kernel](../tutorials/advanced/bring-your-own-kernel.md).

## Evidence

Validate the exact family, backend or kernel, target platform, and task. A
successful package or load test does not establish numerical parity or
performance.
