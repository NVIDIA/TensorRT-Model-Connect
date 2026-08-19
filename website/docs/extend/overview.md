---
title: Contribute & Extend
description: Choose the correct ownership boundary, implement a focused change, and prove it.
---

import Diagram from '@site/src/components/Diagram';

Choose the smallest extension point that matches the change.

<Diagram
  src="/img/diagrams/trtmc-extension-decision.svg"
  alt="Extension decision tree separating native model support, exact-qualified optimized support, shared public changes, and host-supplied runtime dependencies"
  caption="Start with the owning extension path; the required source units and validation evidence follow from that choice."
/>

Model support has two artifact paths inside one ownership tree. Native support
owns a Python `model.py` build recipe, unique `runtime_strategy`, model DSO,
and native E2E JSON manifest in one folder. Exact-qualified optimized support stays inside an existing family
and owns its implementation/profile manifests, isolated adapter, embedded
implementation DSO, Source-side contract tests, and profile semantic-source
digest; it does not need a synthetic native strategy or model DSO. Any
target-hardware qualification is separately retained external evidence.

| Goal | Extension point |
| --- | --- |
| Add native support for a model, even when its task resembles an existing model | [Add a Model Family](add-model-family.md): add one owner folder containing its Python builder, unique runtime strategy/DSO, tests, and native E2E manifest. |
| Add a delegated optimized implementation for an existing family | [Add an Optimized Runtime Implementation](add-optimized-runtime.md): add a family-owned implementation manifest, exact profile, isolated adapter, embedded implementation DSO, semantic-source digest, and Source contract tests. Do not add a synthetic native strategy for it. |
| Add native behavior or another native strategy to an existing model | [Add a Runtime Strategy](add-runtime-strategy.md) under `python/tensorrt_model_connect/models/<owner>/runtime/` and its `MODEL.toml`. |
| Run a new task contract or state model | Extend the public contract only if existing `IPipeline` methods cannot express it, then add the owning model implementation. |
| Add a new user-facing knob | [Add a Config Schema](add-config-schema.md) and consume it in the owning unit. |
| Add a new CLI task | Add a command only when the public task cannot fit an existing command. |
| Add a new verifier | Follow [Validate a Model Contribution](model-validation.md), then add or extend the owning E2E harness plugin, comparator, or reference backend. |

## Cost by kind of change

| Change | Expected ownership |
| --- | --- |
| Another native checkpoint with an identical family contract | Native E2E manifest data and focused evidence. |
| Exact optimized deployment tuple for an existing family | Family-local implementation/profile data, isolated adapter/runtime DSO, and producer qualification evidence. |
| New weight or config variant within a family | Family `model.py` and tests; runtime only when the bundle or request state changes. |
| New graph semantics | Family-local builder/checkpoint logic and parity evidence. |
| New runtime state or operation | Family-owned C++ plugin/pipeline plus C++ and E2E tests. |
| New reusable task contract | E2E runner, comparator, thresholds, and focused evidence. |
| New shared infrastructure | Model-independent shared code plus broad impact proof. |

Before adding a shared abstraction, verify that at least two real owners need
it. Similar implementation does not mean shared runtime identity: every
native runtime strategy maps to exactly one model manifest and one model DSO.
Use E2E `task_strategy` to group different model implementations of the same
user-visible task.

Neither bundle path is a complete operating-system or GPU-runtime image.
Native bundles load the installed model/backend DSOs; optimized bundles embed
their exact implementation DSO. The host still supplies the compatible NVIDIA
driver, CUDA runtime, TensorRT, dynamic loader, and system libraries.

## Contributor path

1. Read the [Contributor Quickstart](contributing.md).
2. Follow the recipe for the owning extension point.
3. Use [Validate a Model Contribution](model-validation.md) when model support
   or model behavior changes.
4. Record exact-revision evidence in the pull request.

The [Developer Guide](../developer-guide/overview.md) and
[Architecture Overview](../architecture/overview.md) explain the units behind
these extension points. Historical migration plans and worklogs are project
records, not current contributor runbooks.

{/* Collaborative review anchor: batch 2. */}
