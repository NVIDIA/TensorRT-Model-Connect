---
title: System Overview
description: The family-owned build, bundle, and runtime boundaries of TensorRT-Model-Connect.
---

import Diagram from '@site/src/components/Diagram';

TensorRT-Model-Connect turns a Hugging Face checkpoint or local model directory
into a TensorRT `.bundle`, then loads that bundle behind an abstract C++ Task
API.

The bundle is the handoff point:

- Python resolves exactly one model family and asks that family's plain
  `model.py::build(request, writer)` function to write the artifact.
- The bundle records only `format`, `family`, `task`, `backend`, and named
  section offsets and lengths.
- C++ loads exactly the backend and family named by the bundle and returns the
  abstract Task interface implemented by that family.

<Diagram
  src="/img/diagrams/trtmc-system-map.svg"
  alt="System map from a checkpoint through one family-owned build into a bundle and then through the runtime loader to an abstract Task API"
  caption="One family owns model behavior end to end; the shared core transfers control at build and load time."
/>

## One path, explicit ownership

There is one build path and one load path. There is no central model registry,
runtime-strategy switch, optimized-provider profile, compatibility layer, or
fallback.

1. The resolver reads standard checkpoint metadata.
2. Dependency-free `families/*/support.py` modules declare ownership and
   supported tasks.
3. Exactly one family must match.
4. Core imports only that family's `model.py` and invokes `build()` once.
5. The family writes its TensorRT engines and assets through `BundleWriter`.
6. At runtime, `load_task()` reads the bounded header, opens the named backend
   and family DSO from an explicit runtime root, and calls the family factory.
7. The application uses the returned abstract interface from `trtmc/task.h`.

## Dependency direction

Applications, examples, and benchmarks depend on public ModelConnect APIs.
Families depend on the narrow build, bundle, Task, and Engine contracts. Core
does not depend on a family, and no family depends on another family.

See [AI-Native Horizontal Scaling Architecture](ai-native-horizontal-scaling.md)
for the complete source and runtime dependency diagrams.

## Source-of-truth entry points

| Boundary | Current implementation |
| --- | --- |
| Build CLI | `core/builder/tensorrt_model_connect/build_cli.py` |
| Public Python build API | `core/builder/tensorrt_model_connect/build.py` |
| Family discovery | `core/builder/tensorrt_model_connect/model_support.py` and `families/*/support.py` |
| Bundle writer | `core/builder/tensorrt_model_connect/bundle_writer.py` |
| Bundle reader | `core/runtime/bundle/` and `core/runtime/include/trtmc/bundle.h` |
| Public Task API | `core/runtime/include/trtmc/task.h` |
| Runtime loader | `core/runtime/loader/family_loader.cpp` |
| TensorRT backends | `core/runtime/tensorrt/` |
| Model implementation | `families/<family>/model.py` and `families/<family>/runtime/` |
| Applications | `apps/` and `examples/` |

## Read next

- [Units and Ownership](units-and-ownership.md)
- [Build Pipeline](build-pipeline.md)
- [Runtime Lifecycle](runtime-lifecycle.md)
- [Bundle Format](bundle-format.md)
- [Validation Design](validation-design.md)
