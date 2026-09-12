---
title: Build System
---

The native build produces small shared mechanics plus independently owned
family DSOs.

| Target | Source owner |
| --- | --- |
| `trtmc_core` | Bundle I/O, device tensors, and engine primitives under `core/runtime/`. |
| `trtmc_runtime` | Exact family/backend loader under `core/runtime/loader/`. |
| `trtmc_backend_trt` | Standard TensorRT Engine implementation. |
| `trtmc_backend_rtx` | Optional TensorRT-RTX Engine implementation. |
| `trtmc` | Native application under `apps/cli/`. |
| `trtmc_benchmark_worker` | Benchmark application under `apps/benchmark/`. |
| `trtmc_model_<family>` | One family's complete native runtime. |

## Family discovery

The root `CMakeLists.txt` glob is limited to
`families/*/runtime/CMakeLists.txt`. Each family file declares its own sources,
private dependencies, warnings, tests, output name, and install rule. Adding a
normal family never changes a central model source list.

The wheel packages `core/builder/tensorrt_model_connect`, the top-level
`families` package, benchmark Python code, and one native product directory at
`tensorrt_model_connect/bin`. The `trtmc` console entry point is a Python
adapter that uses `exec` to replace itself with the native CLI in that
directory; it does not duplicate native files into the wheel scripts area.
Optional family dependencies remain in each
`families/<family>/requirements.txt`; package validation does not import every
family implementation.

CMake generates one private product-build header for all native targets in a
build tree. It is not installed. A release build may set the 32-character
`TRTMC_BUILD_ID` explicitly for coordinated reproducibility; never reuse that
identity across independently compiled native artifact sets.

## Typical source build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=100-real \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_EXAMPLES=OFF
cmake --build build --target \
  trtmc trtmc_backend_trt trtmc_model_qwen
```

Set the CUDA architecture to the selected GPU. Enable
`TRTMC_BUILD_BACKEND_RTX` only with explicit TensorRT-RTX include and library
directories. Disable `TRTMC_ENABLE_BYOK` only when building without the
optional TVM-FFI bridge.

See [Build from Source](../getting-started/source-build.md) for the complete
development-container workflow.
