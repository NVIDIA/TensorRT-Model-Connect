---
title: Source Layout
---

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Canonical Python packaging metadata and build backend declaration. |
| `_pyproject_backend.py` | Local PEP 517/660 wrapper that delegates wheel builds to `conan-py-build` and supports py-only editable installs. |
| `conanfile.py` | Native wheel package recipe used by `conan-py-build`. |
| `CMakeLists.txt` | Native C++/CUDA build entry point. |
| `cmake/` | CMake manifests and generated registration templates. |
| `include/trtmc/` | Public C++ API headers. |
| `src/bundle/` | `.trtfb` bundle reading. |
| `src/cabi/api/` | C ABI entrypoints. |
| `src/runtime/backend/` | Backend DSO loading and backend implementations. |
| `src/runtime/config/` | Schema-driven runtime config. |
| `src/runtime/core/` | Device tensors, caches, samplers, CUDA helpers, schedulers. |
| `src/runtime/domains/` | Modality-specific helpers and generation plans. |
| `src/runtime/models/` | Model runtime folders containing strategy plugins and concrete `IPipeline` implementations. |
| `src/runtime/models/` | Shared runtime plugin helpers. |
| `src/runtime/registry/` | Pipeline factory, registry, and base config parsing. |
| `src/tokenizer/` | Native tokenizers. |
| `python/tensorrt_model_connect/` | Python builder package. |
| `python/tensorrt_model_connect/families/` | Raw TRT family plugins. |
| `tests/builder/` | Python builder tests. |
| `tests/cpp/` | C++ unit tests. |
| `tests/e2e/` | E2E tests and model manifests. |
| `tests/e2e_harness/` | E2E orchestration, runners, references, comparators, thresholds. |
| `tests/tools/` | Tooling tests. |
| `tools/` | Diff, profiling, performance, coverage, and report tools. |
| `scripts/` | Build, cache warmup, E2E scheduling, and scaffolding scripts. |
| `website/` | Docusaurus user documentation site. |
