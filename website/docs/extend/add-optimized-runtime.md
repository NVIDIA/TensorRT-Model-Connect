---
title: Add a Platform-Specific Runtime
description: Keep complete-network platform offload family-owned and explicit.
---

PR #1093 removed the generic optimized-provider/capsule system. The current
public runtime loads a family DSO and either the standard TensorRT backend or
the optional TensorRT-RTX backend. There is no provider registry, profile
matcher, embedded implementation fallback, or compatibility adapter to extend.

Platform specialization is appropriate only for complete-network offload,
such as a TensorRT Edge-LLM integration, or for explicit TVM-FFI BYOK kernel
bindings. Target-specific TensorRT plans, timing caches, or compiled kernels do
not by themselves make a family specialized.

## Ownership rules

If a complete-network platform path is introduced:

1. Keep model selection, checkpoint identity, build orchestration, bundle
   sections, runtime orchestration, and validation in the owning
   `families/<family>/` directory.
2. Do not add a central implementation registry, score/priority matcher,
   silent fallback, or cross-family adapter.
3. Reuse an existing public Task interface; application code must not cast to
   a platform- or family-specific type.
4. Treat the platform toolchain and target environment as explicit external
   dependencies. Pin and record the exact supported combination.
5. Fail once the path claims a request. Do not continue through a different
   backend after a build or load failure.

For TensorRT-RTX, the existing path is backend selection rather than a separate
family implementation:

```bash
python -m tensorrt_model_connect build MODEL \
  --backend trt_rtx \
  -o model-rtx.bundle

trtmc run model-rtx.bundle \
  --runtime-root /opt/trtmc/lib \
  --runtime-cache /tmp/trtmc-rtx.cache \
  --cuda-graphs \
  --prompt "Hello"
```

The selected family must support the request, and the runtime root must contain
`libtrtmc_backend_trt_rtx.so` built against the matching SDK.

## Evidence

Source-contract tests establish only request validation, section/factory
contracts, and fail-closed selection. Target compatibility, parity, and
performance require separately retained evidence tied to the exact source
revision, checkpoint revision, platform/toolchain, hardware, options, inputs,
and artifacts.

The restored
[optimized-runtime design record](../context/optimized-runtime-family-adapter-plan.md)
documents the retired design for historical reference; it is not an extension
API for the current tree.
