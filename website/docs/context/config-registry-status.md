---
title: Central Config Registry - Historical Record
---

:::caution Pre-#1093 architecture history

This page summarizes a configuration-registry experiment from before the
family-owned repository refactor in PR #1093. Its registries, generated schemas,
environment migration, and source paths no longer exist. Do not use this page
as an implementation guide.

:::

## What the experiment tried to solve

The earlier tree accumulated configuration in command-line flags, environment
variables, bundle metadata, and model-specific runtime code. A central,
namespaced registry was explored to make defaults, overrides, validation, and
effective values visible in one place.

The experiment established several useful lessons:

- configuration needs a clear owner and a clear application time;
- build, load, and request settings are different contracts;
- invalid user input should fail where it enters the system;
- model policy should not leak into a central dispatcher; and
- a reproducible command and output matter more than a generated configuration
  framework.

## Why the implementation was retired

The registry required shared schema registration, cross-language generation,
central source lists, and model-specific consumers in shared infrastructure.
That shape conflicted with the new requirement that a normal family change be
contained in one `families/<family>/` directory.

PR #1093 therefore removed the registry instead of carrying a compatibility
layer. There is no schema migration or fallback reader.

## Current configuration boundaries

| Boundary | Current owner |
| --- | --- |
| Checkpoint identity and supported tasks | `families/<family>/support.py` |
| Build inputs | `BuildRequest` plus validation in `families/<family>/model.py` |
| Bundle-wide dispatch identity | the minimal `family`, `task`, and `backend` header |
| Family section schemas | the family that writes and reads those sections |
| Runtime task options | the applicable abstract Task request/config type |
| Backend-only load options | the public load API and TensorRT backend |

Only add a shared configuration field when it is required by an existing
public contract. Otherwise keep the value and its validation with the owning
family.

See [Architecture](../architecture/ai-native-horizontal-scaling.md) and
[Add a Model Family](../extend/add-model-family.md) for the current design.
