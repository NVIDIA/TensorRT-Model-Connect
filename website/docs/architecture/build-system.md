---
title: Build System
---

The repository has a CMake build for native runtime code and a Python packaging flow for builder code.

```mermaid
flowchart TB
  subgraph Native["Native C++/CUDA build"]
    CMake["CMakeLists.txt"] --> Core["trtmc_core"]
    CMake --> CLI["trtmc CLI"]
    CMake --> Backends["backend DSOs"]
    CMake --> Tests["C++ tests"]
  end

  subgraph Python["Python builder package"]
    PyProj["tensorrt_model_connect/pyproject.toml"] --> BuilderModule["tensorrt_model_connect module"]
    BuilderModule --> Families["family plugins"]
    BuilderModule --> Engines["engine builders"]
  end

  subgraph Generated["Generated registration"]
    ManifestPlugins["trtmc_pipeline_plugins.cmake"] --> RegPlugins["register_plugins.cpp"]
    ManifestSchemas["trtmc_config_schemas.cmake"] --> RegSchemas["register_schemas.cpp"]
  end

  RegPlugins --> Core
  RegSchemas --> Core
```

## Native targets

Important CMake targets:

| Target | Purpose |
| --- | --- |
| `trtmc_core` | Core runtime library, public API, registry, plugins, pipelines, tokenizers, CUDA helpers. |
| `trtmc` | CLI executable implemented in `examples/trtmc_cli.cpp`. |
| `trtf_dataset_benchmark` | Dataset benchmark executable. |
| `trtmc_backend_trt` | Standard TensorRT backend DSO when TensorRT headers/libs are available. |
| `trtmc_backend_rtx` | Optional TensorRT-RTX backend target. It outputs `libtrtmc_backend_trt_rtx.so`. |
| `trtmc_tvm_ffi_plugin` | Optional TVM-FFI TensorRT plugin shared library. |

## Link boundaries

`trtmc_core` depends on CUDA runtime and optional CUDA libraries, but standard TensorRT engine execution is behind backend DSOs. The build prints whether the standard TRT backend and TRT-RTX backend are enabled.

The intended boundary is:

```mermaid
flowchart LR
  App["Application or trtmc CLI"] --> Core["trtmc_core<br/>public API, registry, pipelines"]
  Core --> Loader["BackendLoader"]
  Loader --> TrtDso["libtrtmc_backend_trt.so"]
  Loader --> RtxDso["libtrtmc_backend_trt_rtx.so"]
  TrtDso --> LibNvinfer["matching TensorRT runtime"]
  RtxDso --> RtxRuntime["TensorRT-RTX runtime"]
```

The public runtime uses `IBackend` and `ITrtModule` interfaces. TensorRT headers and ABI-sensitive runtime calls stay behind the backend shared objects.

## Python package

`tensorrt_model_connect/pyproject.toml` defines the editable Python builder package consumed by `trtmc build`. Use it for source development with `pip install -e tensorrt_model_connect/`.

The repository-root `pyproject.toml` is the release-wheel build entry point. It uses `conan-py-build` and the root `conanfile.py` to run the native CMake build, then stages only the wheel runtime artifacts under `tensorrt_model_connect/bin/`: the native `trtmc` executable and TensorRT backend DSOs. The release wheel metadata declares TensorRT and the other Python builder dependencies.

The Conan package recipe manages `nlohmann_json` for native wheel builds. TensorRT and CUDA are still supplied by the build environment and by pip/host runtime dependencies rather than by Conan recipes.

To build the release wheel manually, run `python -m build --wheel .` from the repository root with `WHEEL_PYVER`, `WHEEL_ABI`, `WHEEL_ARCH`, and the `TRTMC_TRT_*` / `TRTMC_CUDA_*` paths set. See [Installation](../getting-started/installation.md#build-a-release-wheel) for the full command.

## Generated registration files

CMake uses:

- `cmake/trtmc_pipeline_plugins.cmake`
- `cmake/trtmc_config_schemas.cmake`
- `cmake/trtmc_registration_manifest.cmake`
- `cmake/register_plugins.cpp.in`
- `cmake/register_schemas.cpp.in`

These generated files keep plugin and schema registration declarative.

## Why generated registration exists

Without generated registration, a new runtime strategy would usually require editing a central source file. That creates merge conflicts and makes ownership unclear.

The current design makes registration data-driven:

| Manifest | Consumed by | Result |
| --- | --- | --- |
| `cmake/trtmc_pipeline_plugins.cmake` | CMake configure step | Generated C++ calls to each plugin registrar. |
| `cmake/trtmc_config_schemas.cmake` | CMake configure step | Generated schema registration calls. |
| Plugin source macro | Compiler | A typed function that registers one plugin for one or more strategies. |

This is why extension work normally changes local plugin/schema files plus a manifest entry rather than the factory core.

## Build artifacts to recognize

| Artifact | Meaning |
| --- | --- |
| `build/trtmc` | CLI executable that exercises the public C++ API. |
| `libtrtmc_core.*` | Main runtime library. |
| `libtrtmc_backend_trt.so` | Standard TensorRT backend DSO. |
| `libtrtmc_backend_trt_rtx.so` | Optional TensorRT-RTX backend DSO. |
| `libtrtmc_tvm_ffi_plugin.so` | Optional TensorRT plugin for TVM-FFI kernels. |
| `dist/tensorrt_model_connect-*.whl` | Python wheel containing the builder package, `trtmc` console entry point, packaged native executable, and backend DSOs. |
| `website/build/` | Static Docusaurus docs output. |
