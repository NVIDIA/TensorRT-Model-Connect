---
title: Features
---

Feature support is family-owned. A shared CLI option or Task interface is not a
claim that every family implements it; the exact manifest and family test are
the support contract.

## How to read a page

1. Start with the named family and checkpoint manifest.
2. Separate build-time inputs, load-time inputs, and request-time inputs.
3. Follow the dependency direction: families implement abstract Task and Engine
   contracts; those contracts never depend on a family.
4. Treat skipped hardware validation as untested, not passed.

## Topics

| Topic | Guide |
| --- | --- |
| Family ownership and model discovery | [Model Families](model-families.md) |
| Direct family/task/backend identity | [Runtime Identity](runtime-strategies.md) |
| Explicit build and load options | [Build and Runtime Options](config-and-backends.md) |
| Tensor and context parallel builds | [Multi-Device Execution](multi-device.md) |
| Family-owned precision paths | [Quantization](quantization.md) |
| Qwen sampling behavior | [Top-p Sampling](sampling.md) |
| Historical TriAttention design context | [TriAttention](triattention.md) |
| Direct TVM-FFI kernel integration | [TVM FFI Kernel Bridge](tvm-ffi.md) |

Worked labs live under [Tutorials](../learning-path.md). BYOK, benchmarks, and
examples are one-way applications of public APIs; core and families never
depend on them.

## Evidence boundary

A source inspection can prove ownership and available code paths. A build can
prove TensorRT accepted one graph. A family E2E can prove one exact model and
task. Target support and performance require separate target-hardware evidence.
