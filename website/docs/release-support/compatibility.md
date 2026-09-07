---
title: Compatibility
description: Exact compatibility boundaries for checkpoints, bundles, family DSOs, backends, and target systems.
---

Compatibility is an exact tuple, not a project-wide promise:

```text
checkpoint revision × family build settings × bundle format
× family/runtime build × backend × GPU/SM × driver/CUDA/TensorRT × host
```

| Boundary | Current rule |
| --- | --- |
| Checkpoint | Use the exact model ID or prepared local snapshot declared by the family. A related fine-tune is unverified until tested. |
| Bundle | Format 1 contains `family`, `task`, `backend`, and family-owned sections. Core validates safe bounded reads, not semantic compatibility. |
| Family DSO | Load the exact `libtrtmc_model_<family>.so` produced with the runtime. Family section semantics are private to that owner. |
| Backend DSO | Use the matching TensorRT or TensorRT-RTX backend and compatible runtime libraries. |
| CUDA code | Build family kernels and plugins for the target SM. |
| Multi-device | Match the family-supported TP or CP size and process/device topology. Only families that use collectives load NCCL. |

The project currently makes no cross-release bundle or native ABI compatibility
promise. Core, backend, family DSOs, and CLI should come from the same product
build. Rebuild a bundle after changing the family build contract or TensorRT
cohort.

The generated [Supported Models](../models-recipes/overview.md) inventory shows
what the current source declares. Only an exact target-hardware E2E result
qualifies that tuple.
