---
title: Delegated Runtime Adapter - Retired Design Record
---

:::caution Pre-#1093 architecture history

Before PR #1093, the project experimented with profile-selected delegated
runtime adapters and embedded implementation libraries. That framework,
including its profile descriptors, digest checks, host, and fallback routing,
was removed. This page preserves the design lesson, not an available feature or
qualification claim.

:::

## What the earlier design attempted

The adapter design tried to keep one public bundle/task surface while selecting
a platform-specialized runtime for an exact model, revision, target, and build
configuration. Family-local ownership and fail-closed selection were valuable
goals, but the implementation added another routing system and another artifact
contract beside the native family path.

## Why it was removed

The new architecture has one family build entrypoint and one family runtime
DSO contract. It intentionally has no optimized-runtime registry, profile
selector, embedded implementation host, compatibility adapter, or migration
reader. Once a family build starts, errors are terminal; the core does not try
another family or implementation.

## Current extension choices

Use the smallest existing boundary that matches the need:

- put model behavior and model-specific TensorRT plugins in
  `families/<family>/`;
- use the TensorRT-RTX backend only for its actual backend/runtime behavior;
- use the public BYOK bridge for an explicit external TVM-FFI kernel; or
- add a new public interface only when a real family cannot express its user
  behavior through an existing Task API.

Do not recreate profile selection, artifact digests, or a generic provider
framework around those paths. Platform qualification remains separate evidence
for the exact model, configuration, hardware, and software cohort tested.
