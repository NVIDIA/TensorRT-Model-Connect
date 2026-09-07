---
title: Add Family Configuration
description: Put model policy in the owning family without creating a shared config registry.
---

TensorRT-Model-Connect intentionally has no config-schema registry, layered
config file, `--set` namespace, or environment-variable fallback.

## Build-time configuration

Use an existing typed `BuildRequest` field when it already has the required
cross-family meaning. The owning family's `model.py` validates the value and
either implements it or raises a clear error.

For a family-specific value, prefer data already present in the checkpoint or
a direct constant beside the family implementation. If a user must select it,
add the smallest explicit family-owned input supported by a real end-to-end
case. Do not create an options bag or central registry.

## Runtime configuration

Public request configuration belongs to the relevant abstract Task interface
in `core/runtime/include/trtmc/task.h` only when multiple real families share
the same user-visible meaning. The family pipeline consumes that typed request
directly.

Model-internal scheduling, tensor names, tokenizer settings, and engine order
stay private under `families/<family>/runtime/`.

## Decision test

Before changing shared configuration, answer yes to all of these:

1. Is the requirement used now, not hypothetical?
2. Does it have the same semantics for at least two real families?
3. Can it be expressed as one typed field rather than a map or schema system?
4. Can every affected family either implement or explicitly reject it?
5. Are focused Task/API and family E2E tests included?

If not, keep the choice inside the family.
