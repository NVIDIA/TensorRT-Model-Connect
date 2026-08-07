---
title: Compatibility
description: Compatibility dimensions for bundles, runtimes, models, and target systems.
---

Compatibility is a tuple, not a single project version:

```text
model revision × build configuration × bundle contract × runtime implementation
× GPU/SM × driver/CUDA/TensorRT × host architecture
```

| Boundary | Rule |
| --- | --- |
| Checkpoint | Use the exact `hf_id` and immutable revision retained by the evidence. Same-family fine-tunes are best-effort until tested. |
| Bundle | Inspect compatibility metadata and do not edit sections by hand. |
| Native runtime | Use the matching model/backend DSOs and a compatible TensorRT/CUDA cohort. |
| Platform specialization | Match the exact provider/profile target and options; do not generalize it to native platform support. |
| CUDA code | Compile native extensions and DSOs for the intended SM architecture. |
| Multi-device | Match the topology and world size compiled into the bundle. |

The [Supported Models](../models-recipes/overview.md) page lists declared
checkpoint/configuration contracts. A current target-hardware E2E result is the
evidence required for a verified target claim.
