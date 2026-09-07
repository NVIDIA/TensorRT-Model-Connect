---
title: Build System
description: How core, backends, families, applications, and packages are assembled without a central family list.
---

import Diagram from '@site/src/components/Diagram';

The root build knows conventions, not model names.

<Diagram
  src="/img/diagrams/architecture/build-system.svg"
  alt="CMake builds thin core libraries, discovers family-owned runtime targets by directory convention, and builds applications on top"
  caption="Each family owns its target and sources; applications depend one way on public libraries."
/>

## Native targets

The root `CMakeLists.txt` builds:

- `libtrtmc_core.so` for bundle and device/engine primitives;
- `libtrtmc_runtime.so` for explicit family/backend loading;
- `libtrtmc_backend_trt.so` and the optional TensorRT-RTX backend;
- the `trtmc` CLI and benchmark applications.

CMake discovers `families/*/runtime/CMakeLists.txt` by directory convention.
Each family file defines `libtrtmc_model_<family>.so`, its private sources,
dependencies, warnings, and install rules. Adding a normal family does not edit
a central source list.

## Python package

The wheel contains the builder package under `core/builder`, the `families`
package, benchmark application code, the native CLI, core/runtime/backend
libraries, and every family DSO. Family-owned `requirements.txt` files are
included beside their family.

The root `pyproject.toml` declares only the base environment. A family with
additional build or reference needs declares them locally.

## Applications and examples

`apps/` and `examples/` link or import public ModelConnect APIs. Core,
backends, and families never depend on an application. Optional example targets
are controlled by `TRTMC_BUILD_EXAMPLES`; they do not add behavior to the
library.

## One pinned base image

Development and CI use one pinned base image. A family dependency change does
not publish a new base-image digest. Install the affected family's plain
requirements file before its build or reference tests.

## Minimal source build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90-real
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_qwen
```

See [Build from Source](../getting-started/source-build.md) for the complete
environment flow.
