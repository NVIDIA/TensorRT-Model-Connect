---
title: TriAttention Native Runtime - Historical Worklog
---

:::caution Pre-#1093 architecture history

This is a sanitized summary of the TriAttention bring-up that preceded PR
#1093. Temporary artifact paths, machine-specific commands, environment dumps,
runner details, and point-in-time benchmark numbers were removed. The old
builder/runtime paths no longer exist.

:::

## Purpose

The original work established that TriAttention is a model-owned KV-cache
selection and compaction policy, not merely a replacement attention operator.
The durable design lessons remain relevant even though the repository layout
and runtime contracts changed.

## Lessons retained

### Keep the policy with its family

The family owns the metadata written during build, cache state, score and
selection policy, row compaction, engine bindings, and correctness tests.
Shared core should not interpret TriAttention fields or branch on the family.

### Keep the runtime native

Python is useful for building the TensorRT graph and producing reference
evidence. Production request execution, cache updates, selection, and
compaction remain in the family-owned native DSO.

### Separate logical and physical limits

The built engine's legal tensor shapes, the runtime allocation budget, and the
logical policy budget are different constraints. The family must validate all
three and reject unsupported combinations explicitly.

### Rebind only when shape state changes

A dynamic engine can continue after compaction when the family updates the
live cache shape and bindings consistently. Avoid rebuilding or switching
engines unless a concrete profile contract requires it.

### Compare against a fair control

Correctness and performance evidence must use the same checkpoint, prompts,
sampling policy, token budget, engine family, hardware, and timing boundary.
Small pilot results are debugging evidence, not a broad performance claim.

## Current ownership

The current implementation is owned by the applicable family under
`families/`, including its build code, runtime cache implementation, manifests,
and tests. Runtime-sized KV memory enters through the public load boundary;
core does not own the family policy.

Consult the live family source and tests for supported settings. Do not copy
commands or numbers from the pre-refactor worklog into current support claims.
