---
title: Units and Ownership
description: Source-level responsibilities and the authoritative contracts between units.
---

import Diagram from '@site/src/components/Diagram';

Architecture explains the system flow. Unit ownership answers a different
question: **where does a behavior belong?**

The repository follows one rule throughout the build, runtime, and test trees:
model-specific knowledge stays with the model that owns it. Shared units own
public contracts, bundle transport, loading, configuration mechanics, device
abstractions, and code that is genuinely model-independent.

## Ownership block diagram

<Diagram
  src="/img/diagrams/architecture/build-ownership.svg"
  alt="Build-time ownership from public CLI and Python entry points through a model family to native or optimized bundle construction"
  caption="Build-time model knowledge stays in the owning family; the bundle is the only artifact passed into runtime."
/>

<Diagram
  src="/img/diagrams/architecture/runtime-ownership.svg"
  alt="Runtime ownership from the trtmc CLI, Python Pipeline wrapper, C++ API, and limited C-linkage surface through shared loaders to an optimized implementation or native model and backend DSOs"
  caption="All public runtime surfaces enter shared factory and loader units; optimized artifacts own a private implementation, while native model and backend DSOs divide task behavior from engine execution."
/>

The backend arrow is an interface boundary. A model DSO does not select and
link a backend directly; `PipelineFactory` loads a compatible backend and
passes its `IBackend*` through `PipelineContext`.

## Public authorities

Use these files as the contract authority. Implementation pages and examples
must agree with them.

| Contract | Authority |
| --- | --- |
| Pipeline operations, request configs, and typed results | `include/trtmc/pipeline.h` |
| Bundle inspection types | `include/trtmc/bundle.h` |
| Pipeline factory and native pool construction | `include/trtmc/runtime/pipeline_factory.h` |
| Native plugin construction context | `include/trtmc/runtime/pipeline_plugin.h` |
| Native strategy registry | `include/trtmc/runtime/pipeline_registry.h` |
| Model DSO loading | `include/trtmc/runtime/pipeline_plugin_loader.h` |
| Backend and engine-execution abstraction | `include/trtmc/runtime/trt_backend.h`, `include/trtmc/runtime/trt_module.h` |
| Runtime configuration | `include/trtmc/config/` |
| Python family build entry | `python/tensorrt_model_connect/families/<family>/model.py` |
| Python bundle serialization | `python/tensorrt_model_connect/bundle_writer.py` |

The declarations with C linkage at the bottom of `pipeline.h` are a limited
C-linkage C++ subset. They expose C++ types and do not provide a complete
opaque-handle ownership API, so they are not a stable pure-C ABI.

## Major unit groups

| Unit group | Main path | Owns |
| --- | --- | --- |
| Public C++ API | `include/trtmc/` | Task methods, typed results, load options, bundle inspection |
| CLI | `src/cli/` | Argument parsing, file/media adaptation, and calls into public APIs |
| Python package | `python/tensorrt_model_connect/` | Public build API, CLI bridge, family resolution, and bundle writing |
| Python families | `python/tensorrt_model_connect/families/<family>/` | Checkpoint/config semantics, native builders, and optional optimized implementations |
| Bundle implementation | `src/bundle/` | Safe header parsing and section access |
| Runtime registry | `src/runtime/registry/` | Native strategy resolution, DSO loading, plugin lookup, and pipeline creation |
| Optimized host | `src/runtime/providers/` | Descriptor validation, artifact materialization, and private factory loading |
| Model runtimes | `src/runtime/models/<owner>/` | Native plugins, pipelines, state, samplers, preprocessors, and model CUDA |
| Runtime core | `src/runtime/core/` | Device tensors, streams, pooling, distributed setup, and reusable execution mechanics |
| Runtime domains | `src/runtime/domains/` | Small assumption-free helpers shared by real model owners |
| Backend | `src/runtime/backend/` | TensorRT ABI-sensitive engine deserialization and execution |
| Tests and evidence | `tests/` | Builder, C++, tool, E2E, and qualification contracts |

## Descriptor relationship

Three native descriptor roots own different parts of one supported model:

| Descriptor | Responsibility |
| --- | --- |
| `python/tensorrt_model_connect/families/<family>/MODEL.toml` | Builder discovery and specialization metadata |
| `src/runtime/models/<owner>/MODEL.toml` | Native model DSO, registrar symbols, runtime strategies, config schemas, and focused C++ tests |
| `tests/e2e/models/<family>/MODEL.toml` | E2E manifests, task defaults, and validation ownership |

The IDs normally align, but `runtime_strategy` is the authoritative link when a
builder/E2E family and runtime owner have different compatibility names.
`task_strategy` is separate: it selects the user-task runner/comparator
contract, not a native runtime DSO.

An optimized implementation uses a different ownership shape:

```text
python/tensorrt_model_connect/families/<family>/<implementation>/IMPLEMENTATION.toml
python/tensorrt_model_connect/families/<family>/<implementation>/profiles/*.toml
tests/e2e/models/<family>/<implementation>/test_adapter.py
tests/e2e/models/<family>/<implementation>/test_runtime_contract.py
```

It produces an embedded implementation DSO and does not need a synthetic native
strategy or runtime model descriptor. Source-side tests prove selection,
packaging, and fail-closed runtime contracts. The profile's semantic-source
digest and separately retained external target evidence carry different
meanings; neither should be presented as the other.

## Boundaries that prevent accidental coupling

| Boundary | Reason |
| --- | --- |
| Build adapter versus runtime constructor | Build code understands source artifacts; runtime code understands deployment artifacts and requests. |
| `IPipelinePlugin` versus `IPipeline` | The plugin validates and constructs once; the pipeline owns request-time behavior. |
| Model DSO versus backend DSO | The model owns semantics; the backend owns TensorRT ABI and engine execution. |
| `runtime_strategy` versus `task_strategy` | One dispatches a native implementation; the other groups an E2E user contract. |
| Schema config versus one-off flags | Schemas provide types, defaults, validation, provenance, and shared CLI/API surfaces. |
| Unit evidence versus model qualification | Local tests isolate behavior; exact checkpoint/hardware evidence proves integration or performance. |

## Reading a change

When tracing an unfamiliar change, follow its owner before following similar
code elsewhere:

1. Find the Python, runtime, or E2E descriptor that declares the owner.
2. Follow the exact strategy, implementation/profile, or task identity named by
   that descriptor.
3. Read the family `model.py`, runtime plugin, or optimized adapter before
   reading shared infrastructure.
4. Check which test or qualification artifact proves the stated behavior.
5. Escalate to a shared abstraction only when multiple independent owners need
   the same assumption-free contract.

For implementation recipes, continue to [Extend the Project](../extend/overview.md).

{/* Collaborative review anchor: batch 2. */}
