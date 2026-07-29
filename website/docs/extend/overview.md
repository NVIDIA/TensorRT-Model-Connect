---
title: Extend the Project
---

Choose the smallest extension point that matches the change.

![Extension decision tree separating native model support, exact-qualified optimized support, shared public changes, and host-supplied runtime dependencies](/img/diagrams/trtmc-extension-decision.svg)

Model support has two distinct ownership paths. Native support owns a
`FamilyPlugin`, unique `runtime_strategy`, model DSO, and native E2E JSON
manifest. Exact-qualified optimized support stays inside an existing family
and owns its implementation/profile manifests, isolated adapter, embedded
implementation DSO, and qualification TOML; it does not need a synthetic
native strategy or model DSO.

| Goal | Extension point |
| --- | --- |
| Add native support for a model, even when its task resembles an existing model | Add a Python family package, a unique model-owned runtime strategy/DSO, and a native E2E JSON manifest. |
| Add a delegated optimized implementation for an existing family | Add a family-owned implementation manifest, exact-qualified profile, isolated adapter, embedded implementation DSO, and producer qualification TOML. Do not add a synthetic native strategy for it. |
| Add native behavior or another native strategy to an existing model | Extend `src/runtime/models/<owner>/` and its `MODEL.toml`. |
| Run a new task contract or state model | Extend the public contract only if existing `IPipeline` methods cannot express it, then add the owning model implementation. |
| Add a new user-facing knob | Add a config schema and consume it in the owning unit. |
| Add a new CLI task | Add a command only when the public task cannot fit an existing command. |
| Add a new verifier | Add or extend an E2E harness plugin, comparator, or reference backend. |

Before adding a shared abstraction, verify that at least two real owners need
it. Similar implementation does not mean shared runtime identity: every
native runtime strategy maps to exactly one model manifest and one model DSO.
Use E2E `task_strategy` to group different model implementations of the same
user-visible task.

Neither bundle path is a complete operating-system or GPU-runtime image.
Native bundles load the installed model/backend DSOs; optimized bundles embed
their exact implementation DSO. The host still supplies the compatible NVIDIA
driver, CUDA runtime, TensorRT, dynamic loader, and system libraries.
