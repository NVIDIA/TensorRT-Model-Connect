---
title: Add Family Runtime Behavior
description: Implement runtime behavior inside one family DSO.
---

The current architecture has no runtime-strategy registry. A bundle names one
family and task, and the loader opens exactly
`libtrtmc_model_<family>.so`.

## Implement the task

1. Choose an existing abstract interface in
   `core/runtime/include/trtmc/task.h`.
2. Implement the concrete pipeline under `families/<family>/runtime/`.
3. Keep tokenizer, preprocessing, execution order, state, sampling, and
   postprocessing in that family.
4. Create engines through the abstract `IBackend`/Engine API.
5. Export the family factory expected by
   `core/runtime/include/trtmc/runtime/family_factory.h`.
6. Return an `ITask` whose `task()` matches the bundle header.

If no public Task interface represents the required user behavior, add the
smallest typed interface only after proving the current cross-family need. Do
not expose family classes to applications.

## Build the DSO

The family-owned `runtime/CMakeLists.txt` defines
`trtmc_model_<family>`, all private sources, and direct dependencies. Root
CMake discovers this file by convention, so no central source list changes.

Load NCCL only when the family actually executes collectives. Rank-specific
section selection alone does not require a communicator.

## Failure behavior

Missing sections, unsupported options, backend errors, and task mismatches are
terminal. Do not fall back to another family, Python runtime, or alternate
pipeline.
