---
title: Units and Ownership
---

The repository uses one primary ownership rule: model-specific behavior stays
inside one `families/<family>/` vertical slice. Similar code is deliberately
duplicated across families until a stable model-agnostic contract exists.

## Public contracts

| Contract | Location | Owns |
| --- | --- | --- |
| Python build request and entry point | `core/builder/tensorrt_model_connect/build.py` | Resolved build inputs and one control transfer to a family. |
| Model support protocol | `core/builder/tensorrt_model_connect/model_support.py` | Dependency-free exact checkpoint matching. |
| Bundle writer | `core/builder/tensorrt_model_connect/bundle_writer.py` | Format-1 header and streaming named sections. |
| Task API | `core/runtime/include/trtmc/task.h` | User-visible task requests and results. |
| Bundle API | `core/runtime/include/trtmc/bundle.h` | Bounded immutable bundle inspection and section reads. |
| Family factory and loader | `core/runtime/include/trtmc/runtime/family_factory.h`, `family_loader.h` | One family/backend control transfer. |
| Engine API | `core/runtime/include/trtmc/runtime/trt_backend.h`, `trt_module.h` | Backend-neutral engine creation, binding, and enqueue. |
| BYOK API | `core/runtime/include/trtmc/byok.h` | Explicit TVM-FFI kernel binding. |

## Physical units

| Unit | Ownership boundary |
| --- | --- |
| `families/<family>/` | All topology, weight mapping, bundle-section semantics, native pipeline, model dependencies, fixtures, manifests, thresholds, and oracles for one family. |
| `core/builder/` | Model-agnostic Python build contracts, discovery, bundle mechanics, and unit tests. |
| `core/runtime/bundle/` and `primitives/` | `libtrtmc_core.so` bundle, tensor, CUDA, and engine primitives. |
| `core/runtime/loader/` | `libtrtmc_runtime.so` exact family/backend loading. |
| `core/runtime/tensorrt/` | Standard TensorRT and optional TensorRT-RTX Engine implementations. |
| `apps/cli/` | Native command parsing and private image/WAV/file adapters. |
| `apps/benchmark/` | Benchmark catalog, workers, reference runners, and reporting policy. |
| `examples/` | Optional applications over public APIs. |

## Allowed dependency direction

```text
applications -> public build/load/Task/BYOK contracts
family build -> BuildRequest + BundleWriter + TensorRT build API
family runtime -> BundleReader + Task + Engine contracts
runtime loader -> bundle + factory + backend contracts
TensorRT backend -> Engine contract
```

The always-forbidden edges are family-to-family dependencies, core-to-concrete
family dependencies, backend-to-family dependencies, and family/core
dependencies on applications. A family-local custom TensorRT plugin is allowed
only when the family graph genuinely requires it; TVM-FFI BYOK remains the
model-agnostic custom-kernel boundary.
