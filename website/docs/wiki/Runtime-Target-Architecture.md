# Runtime Target Architecture

Status: implemented, with model-owned dynamic DSOs.

## Invariants

- A new runtime family lives below `src/runtime/models/<family>/`.
- Its `MODEL.toml` declares the model DSO, registration symbols, runtime
  strategies, config schemas, and C++ tests.
- Every strategy is unique and normally family-qualified.
- CMake discovers all runtime descriptors and creates one model target per
  descriptor.
- `PipelineFactory` resolves strategy to DSO at runtime and does not link every
  model implementation into the core.
- Model-specific pipeline/state behavior remains in the model folder.
- Public C and C++ callers interact through stable public headers and do not
  depend on model-private classes.

## Runtime directory roles

| Path | Responsibility |
| --- | --- |
| `include/trtmc/runtime/` | Public factory, registry, plugin, tensor, tokenizer, and backend contracts |
| `src/runtime/registry/` | Bundle materialization, DSO loading, registry, factory |
| `src/runtime/config/` | Schema registration and layered configuration |
| `src/runtime/core/` | Model-independent runtime primitives |
| `src/runtime/domains/` | Small cross-model modality helpers |
| `src/runtime/models/<family>/` | Concrete model DSO and pipeline behavior |

Generic strings such as `decoder_kv_cache`, `decoder_moe`,
`vision_language`, or `encoder_only` are not the current strategy inventory.
Use `src/runtime/models/*/MODEL.toml` for the live keys.

## Build-time discovery

`cmake/trtmc_pipeline_plugins.cmake` scans
`src/runtime/models/*/MODEL.toml`, validates each descriptor, builds the
declared plugin sources, and generates runtime-manifest metadata. A contributor
adds a runtime by adding its model-owned descriptor and sources, not by
editing a registration table in the factory.

## Runtime resolution

The bundle must identify the runtime strategy. The loader resolves the owning
library, loads it, and looks up the strategy in `PipelineRegistry`. Failure to
load the DSO or find the strategy is an error; the runtime does not silently
select an unrelated generic pipeline.
