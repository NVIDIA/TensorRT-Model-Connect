---
title: Build System
description: Native targets, DSO boundaries, generated registration, and Python packaging.
---

import Diagram from '@site/src/components/Diagram';

The repository combines:

- CMake for C++/CUDA libraries, executables, model DSOs, backend DSOs, and C++
  tests;
- the root `pyproject.toml` and `conanfile.py` for Python packaging and release
  wheels; and
- Docusaurus under `website/` for this documentation site.

Model implementation routing is explained in
[Build Pipeline](build-pipeline.md). This page focuses on how code and artifacts
are assembled.

## Native targets

| Target | Purpose |
| --- | --- |
| `trtmc_core` | Public API, bundle/config handling, registries, loaders, and shared runtime mechanics |
| `trtmc` | Command-line executable under `src/cli/` |
| `trtmc_model_plugins` | Aggregate target for manifest-discovered native model DSOs |
| `trtmc_model_<owner>` | One model-owned runtime DSO under `build/models/<owner>/` |
| `trtmc_backend_trt` | Standard TensorRT backend DSO when its SDK is available |
| `trtmc_backend_rtx` | Optional TensorRT-RTX backend, emitted as `libtrtmc_backend_trt_rtx.so` |
| `trtmc_tvm_ffi_plugin` | Optional TensorRT plugin for trusted TVM-FFI kernels |
| `trtmc_dataset_benchmark` | Dataset benchmark executable when benchmarks are enabled |
| `trtmc_benchmark_worker` | Worker used by benchmark tooling |

Optimized implementations are not `trtmc_model_<owner>` targets. A selected
family adapter supplies an exact `libtrtmc_impl_*.so` and embeds it in the
optimized bundle.

## Native link boundary

<Diagram
  src="/img/diagrams/architecture/build-system.svg"
  alt="Runtime library graph from an application through trtmc_core to a model DSO, backend loader, TensorRT backend DSO, stable execution interfaces, and matching runtime"
  caption="The shared core loads model and backend DSOs independently; their stable IBackend and ITrtModule interfaces prevent a model implementation from owning backend selection."
/>

`trtmc_core` loads a backend and injects its `IBackend*` into a model plugin's
`PipelineContext`. Model DSOs program against public interfaces; they do not
directly choose a backend DSO.

TensorRT headers and ABI-sensitive engine execution remain behind backend
libraries. Model-owned pipeline, state, pre/postprocessing, and CUDA code remain
behind model libraries.

## Manifest-generated native registration

CMake scans `src/runtime/models/*/MODEL.toml`. Each descriptor declares:

- its model ID and output library;
- plugin source/registrar pairs;
- unique native runtime strategies;
- optional model-owned config schemas; and
- focused C++ tests.

The configure step validates those declarations and generates:

- a strategy-to-model/library index linked into `trtmc_core`;
- one exported registrar translation unit for each model DSO; and
- shared/model schema registration calls.

Primary generator inputs are:

- `cmake/trtmc_pipeline_plugins.cmake`;
- `cmake/model_plugin_index.cpp.in`;
- `cmake/register_model_plugin.cpp.in`;
- `cmake/trtmc_config_schemas.cmake`;
- `cmake/trtmc_registration_manifest.cmake`;
- `cmake/register_schemas.cpp.in`.

This is why adding a native model does not require editing a central switch in
`PipelineFactory` or a hand-maintained target list.

Optimized implementation/profile discovery is Python-family-owned and does not
consume this native index.

## Python package

The root `pyproject.toml` is the Python packaging entry point. Package source
lives under `python/tensorrt_model_connect/`.

Two installation shapes serve different purposes:

| Shape | Behavior |
| --- | --- |
| `pip install -e . -C py-only=true` | Developer-only editable Python package; does not run CMake/Conan or install native artifacts |
| Release wheel | Builds/stages Python builder code plus the native CLI, core library, backend DSOs, benchmark worker, and model DSOs |

Use the Python-only editable install with a separate source-tree CMake build
when developing:

```bash
pip install -e . -C py-only=true
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Release wheels use `conan-py-build` and the root `conanfile.py` to run the
native CMake build, then stage runtime artifacts in a `bin/` subdirectory of
the installed Python package: the native `trtmc` executable, `libtrtmc_core`,
TensorRT backend DSOs, the benchmark worker, and all model DSOs. The same
native executable is also staged into the wheel scripts directory so pip
installs `trtmc` into the target environment's `bin/` directory. Release-wheel
metadata declares TensorRT and the other Python builder dependencies.

The Conan recipe manages `nlohmann_json` for native wheel builds. TensorRT and
CUDA are supplied by the build environment and by pip/host runtime dependencies,
not by Conan recipes. Release wheel builds also disable the optional
libtorch-backed multinomial sampler so the wheel does not link against
PyTorch's native DSOs or inherit their platform floor.

Release validation uses the repository `Dockerfile`, pinned to the official
TensorRT 11.1 CUDA 13 cohort on Ubuntu 24.04 / glibc 2.39. `auditwheel` verifies
the `manylinux_2_39_aarch64` tag. Package validation builds and installs the
wheel; source tests configure, build, and test the exact source revision
separately.

To build the release wheel manually, run `python -m build --wheel .` from the
repository root with `WHEEL_PYVER`, `WHEEL_ABI`, `WHEEL_ARCH`, and the
`TRTMC_TRT_*` / `TRTMC_CUDA_*` paths set. See
[Installation](../getting-started/installation.md#2-build-the-aarch64-wheel-from-source)
for the full command.

## Artifacts to recognize

| Artifact | Meaning |
| --- | --- |
| `build/trtmc` | Source-built CLI executable |
| `libtrtmc_core.*` | Shared public runtime |
| `libtrtmc_backend_trt.so` | Standard TensorRT backend |
| `libtrtmc_backend_trt_rtx.so` | Optional TensorRT-RTX backend |
| `build/models/<owner>/libtrtmc_model_<owner>.so` | Model-owned native runtime |
| `libtrtmc_tvm_ffi_plugin.so` | Optional TVM-FFI TensorRT plugin |
| `dist/tensorrt_model_connect-*.whl` | Built Python/native wheel |
| Native `.trtfb` | Plans/assets dispatched through an installed model/backend DSO |
| Optimized `.trtfb` | Descriptor plus embedded implementation DSO/artifact tree |
| `website/build/` | Docusaurus production output |

## Build-time versus run-time availability

A configured target proves only that its dependencies and source were available
to that build. Run-time success additionally depends on:

- the bundle's required strategy or implementation identity;
- discoverable native DSOs or valid embedded optimized artifacts;
- compatible TensorRT/CUDA/driver libraries;
- legal optimization-profile shapes; and
- the selected model's task contract.

Use [Validation Design](validation-design.md) to choose evidence beyond
successful compilation.
