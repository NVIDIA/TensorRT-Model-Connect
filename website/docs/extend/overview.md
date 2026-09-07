---
title: Contribute & Extend
description: Choose the owning vertical slice, implement a focused change, and prove it.
---

Choose the smallest current ownership boundary that matches the change.

| Goal | Extension point |
| --- | --- |
| Add support for a checkpoint family | [Add a Model Family](add-model-family.md): add one complete `families/<family>/` vertical slice. |
| Add behavior to an existing family | Edit that family's `model.py`, `runtime/`, and family-owned tests. There is no separate runtime-strategy registry. |
| Add a stable user-visible task | Extend `core/runtime/include/trtmc/task.h` only when no existing Task interface can express it, then implement it inside the owning family. |
| Add a family-only build option | Keep it inside that family when it can be derived from checkpoint identity, task, or existing request fields. |
| Add a truly model-agnostic build/load option | Follow [Configuration Boundaries](add-config-schema.md) and change the narrow shared request or Task contract with broad tests. |
| Add complete-network platform offload | Follow [Platform-Specific Runtime](add-optimized-runtime.md); keep model policy in the family and avoid a central provider/fallback system. |
| Bind a custom kernel | Use the public [TVM-FFI BYOK](../features/tvm-ffi.md) graph-transform and runtime binding contracts. |
| Add a benchmark or hardware example | Implement an application under `apps/benchmark/` or `examples/` over public APIs. |

## Cost by kind of change

| Change | Expected ownership |
| --- | --- |
| Another exact checkpoint with the same family contract | Family `support.py`, manifests, and focused evidence. |
| New weight/config variant | Family builder and tests; runtime only when sections or Task behavior change. |
| New graph semantics | Family-local TensorRT graph and weight mapping plus parity evidence. |
| New runtime state or operation | Family runtime implementation plus C++ and E2E tests. |
| New reusable Task contract | Narrow shared header change, at least one family implementation, CLI/application coverage, and broad validation. |
| New shared mechanism | Model-agnostic core only, with proof that it contains no topology, policy, or family behavior. |

Similar implementations do not justify a cross-family abstraction. Each team
must be able to implement, validate, change, and revert its family without
editing or coordinating with siblings.

## Contributor path

1. Read [Contributing](contributing.md).
2. Follow the guide for the owning extension point.
3. Use [Validate a Model Contribution](model-validation.md) for family changes.
4. Record exact source, checkpoint, hardware, commands, and retained evidence
   in the pull request.

The [Architecture Overview](../architecture/overview.md) explains the current
units. Restored context documents and migration worklogs are historical records,
not current contributor runbooks.
