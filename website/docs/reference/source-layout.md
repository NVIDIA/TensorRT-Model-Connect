---
title: Source Layout
---

| Path | Purpose |
| --- | --- |
| `include/trtmc/` | Public C++ API headers. |
| `src/bundle/` | `.trtfb` bundle reading. |
| `src/cabi/api/` | C ABI entrypoints. |
| `src/runtime/backend/` | Backend DSO loading and backend implementations. |
| `src/runtime/config/` | Schema-driven runtime config. |
| `src/runtime/core/` | Device tensors, caches, samplers, CUDA helpers, schedulers. |
| `src/runtime/domains/` | Modality-specific helpers and generation plans. |
| `src/runtime/plugins/` | Runtime strategy plugins. |
| `src/runtime/models/` | Concrete model-family pipeline implementations. |
| `src/runtime/registry/` | Pipeline factory, registry, and base config parsing. |
| `src/tokenizer/` | Native tokenizers. |
| `tensorrt_model_connect/tensorrt_model_connect/` | Python builder package. |
| `tensorrt_model_connect/tensorrt_model_connect/families/` | Raw TRT family plugins. |
| `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/` | Torch-TRT engine definition path. |
| `tests/builder/` | Python builder tests. |
| `tests/cpp/` | C++ unit tests. |
| `tests/e2e/` | E2E tests and model manifests. |
| `tests/e2e_harness/` | E2E orchestration, runners, references, comparators, thresholds. |
| `tests/tools/` | Tooling tests. |
| `tools/` | Diff, profiling, performance, coverage, and report tools. |
| `scripts/` | Build, cache warmup, E2E scheduling, and scaffolding scripts. |
| `cmake/` | CMake manifests and generated registration templates. |
| `website/` | Docusaurus user documentation site. |
