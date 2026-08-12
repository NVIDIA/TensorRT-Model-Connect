---
title: Developer Guide
description: Understand ownership boundaries, change the implementation, and contribute a validated vertical slice.
---

The Developer Guide is for readers who need to understand or modify the source
tree. It separates two questions:

- **Architecture:** which component owns checkpoint resolution, TensorRT graph
  construction, bundle format, runtime dispatch, task execution, and evidence?
- **Extension:** which files and tests form the smallest complete change for a
  model family, runtime strategy, optimized provider, or config schema?

## Read by change type

| Change | Start here | Then follow |
| --- | --- | --- |
| Understand the system | [Architecture Overview](../architecture/overview.md) | Units, build pipeline, runtime lifecycle, validation design |
| Add a model | [Add a Model Family](../extend/add-model-family.md) | Model-owned Python, C++ DSO, manifest, and E2E proof |
| Add native runtime behavior | [Add a Runtime Strategy](../extend/add-runtime-strategy.md) | Unique strategy ownership and runtime registration |
| Add platform specialization | [Add an Optimized Runtime](../extend/add-optimized-runtime.md) | Exact profile, provider adapter, implementation DSO, and separate qualification |
| Add user-facing configuration | [Add a Config Schema](../extend/add-config-schema.md) | Matching Python/C++ schema plus the owning consumer |
| Submit a contribution | [Contributor Quickstart](../extend/contributing.md) | Focused validation and PR evidence |

User-facing task instructions belong in [User Guides](../user-guides/overview.md).
Progressive labs belong in [Tutorials](../learning-path.md). Keep architecture
and contribution mechanics here so neither path becomes a prerequisite for a
normal user.

## General runtime strategy

For one exact `model × platform × configuration` tuple, Model Connect selects
one runtime owner. It can be the model family's native TensorRT runtime or an
exact platform-specialized provider. Selection must not create two competing
owners for the same qualified tuple or silently fall through after one provider
claims it.

The native and platform-specialized paths are separate implementation
boundaries. Users normally interact with the same bundle/task API; advanced
developers use the provider/profile metadata and runtime lifecycle pages to
understand which backend owns execution.

{/* Collaborative review anchor: batch 2. */}
