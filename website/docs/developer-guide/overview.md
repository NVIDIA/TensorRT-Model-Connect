---
title: Developer Guide
description: Understand ownership boundaries, change the implementation, and contribute a validated vertical slice.
---

The Developer Guide is for readers who need to understand or modify the source
tree. It separates two questions:

- **Architecture:** which component owns checkpoint resolution, TensorRT graph
  construction, bundle format, runtime dispatch, task execution, and evidence?
- **Extension:** which family-local files and tests form the smallest complete
  change for a model family, native runtime, platform offload, or typed input?

## Read by change type

| Change | Start here | Then follow |
| --- | --- | --- |
| Understand the system | [Architecture Overview](../architecture/overview.md) | Units, build pipeline, runtime lifecycle, validation design |
| Add a model | [Add a Model Family](../extend/add-model-family.md) | Model-owned Python, C++ DSO, manifest, and E2E proof |
| Add native runtime behavior | [Extend a Family Runtime](../extend/add-runtime-strategy.md) | Family-local task implementation, build target, and evidence |
| Add platform specialization | [Platform Runtime Extensions](../extend/add-optimized-runtime.md) | Complete-network family offload and separate qualification |
| Add user-facing configuration | [Configuration Boundaries](../extend/add-config-schema.md) | Explicit build, load, or typed request field plus its owner |
| Submit a contribution | [Contributor Quickstart](../extend/contributing.md) | Focused validation and PR evidence |

User-facing task instructions belong in [User Guides](../user-guides/overview.md).
Progressive labs belong in [Tutorials](../learning-path.md). Keep architecture
and contribution mechanics here so neither path becomes a prerequisite for a
normal user.

## Runtime ownership

Each family owns its native task implementation and orchestration. Shared code
loads the installed family DSO and exposes model-agnostic task contracts; it
does not select through a central runtime-strategy or provider registry.
Complete-network platform offload remains family-owned and must not create a
second generic orchestration owner.

{/* Collaborative review anchor: batch 2. */}
