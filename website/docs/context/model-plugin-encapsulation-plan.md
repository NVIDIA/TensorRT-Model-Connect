---
title: Family Isolation - Design History
---

:::caution Pre-#1093 architecture history

This record captures the reasoning that led to PR #1093. The old split
builder/runtime/test roots and descriptor files have been removed. Paths and
commands below describe only the current result.

:::

## Goal

A model family is one complete, independently changeable vertical slice.
Adding or repairing a normal family should not require edits to core, sibling
families, a central registry, or a central source list.

The current physical unit is:

```text
families/<family>/
├── support.py
├── model.py
├── requirements.txt        # optional
├── runtime/
└── tests/
```

## Decisions retained from the migration

1. Model-specific build, runtime, preprocessing, postprocessing, and tests stay
   in the owning family.
2. Families never import, include, link, or read sibling-family files.
3. A Python family builder is a plain `build(request, writer)` function. It
   must not inherit from a base builder.
4. A runtime family DSO implements an abstract Task interface and drives
   engines through the abstract Engine API.
5. Similar code may be duplicated. Similarity alone is not evidence for a
   shared model abstraction.
6. Each family owns its manifests, oracle, thresholds, fixtures, and focused
   tests.
7. Shared infrastructure is limited to discovery and build contracts, bundle
   I/O, abstract Task and Engine APIs, loader mechanics, stable device
   primitives, and the exercised BYOK bridge.

## Control transfer

The shared layers transfer control only twice:

```text
checkpoint metadata
  -> resolver calls dependency-free family support declarations
  -> build core imports one selected family/model.py
  -> family writes a bundle

bundle
  -> runtime loader opens the named family and backend DSOs
  -> family factory returns a concrete abstract-Task implementation
  -> application calls that Task directly
```

Runtime loading does not create a source dependency from core to a concrete
family. The family depends on and implements the abstract contract.

## Acceptance boundary

A family is complete only when it resolves its supported checkpoint, builds a
real bundle, loads with sibling DSOs absent, performs a real Task call, and
passes its model-owned correctness criteria. Static discovery or compilation
alone is not a support claim.

For current implementation instructions, use
[Add a Model Family](../extend/add-model-family.md). For the full dependency
rules, use the
[AI-Native Horizontal Scaling Architecture](../architecture/ai-native-horizontal-scaling.md).
