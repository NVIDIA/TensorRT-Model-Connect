# Runtime Target Architecture

Status: implemented native target architecture. Optimized-runtime bundles use
the separate embedded implementation-DSO path described below.

## Native-path invariants

- A new runtime family lives below `src/runtime/models/<family>/`.
- Its `MODEL.toml` declares the model DSO, registration symbols, runtime
  strategies, config schemas, and C++ tests.
- Every strategy is unique and normally family-qualified.
- CMake discovers all runtime descriptors and creates one model target per
  descriptor.
- For native bundles, `PipelineFactory` resolves strategy to DSO at runtime and
  does not link every model implementation into the core.
- Model-specific pipeline/state behavior remains in the model folder.
- C++ callers use the public C++ headers and do not depend on model-private
  classes. The current C-linkage subset still exposes C++ types and lacks a
  pipeline-destroy operation, so it is not a C-compatible public header or a
  stable, complete C ABI. C-facing consumers need a C++ shim.

## Exact-qualified optimized-path invariants

- Optimized selection is bounded to an existing Python family and requires an
  exact implementation/profile match for the model, revision, target, and
  build options.
- The family-owned adapter runs in isolation and packages
  `optimized_runtime.json`, provider-produced artifacts, and the exact embedded
  `libtrtmc_impl_*.so`.
- This path does not require a synthetic native `runtime_strategy`,
  `src/runtime/models/<family>/MODEL.toml`, model DSO, or backend DSO.
- Zero qualified claims continue to the native builder. Multiple claims are an
  error, and failure after one adapter is selected is terminal.

## Runtime directory roles

| Path | Responsibility |
| --- | --- |
| `include/trtmc/runtime/` | Public factory, registry, plugin, tensor, tokenizer, and backend contracts |
| `src/runtime/registry/` | Bundle materialization, DSO loading, registry, factory |
| `src/runtime/config/` | Schema registration and layered configuration |
| `src/runtime/core/` | Model-independent runtime primitives |
| `src/runtime/domains/` | Small cross-model modality helpers |
| `src/runtime/models/<family>/` | Concrete model DSO and pipeline behavior |
| `src/runtime/providers/` | Generic optimized-runtime descriptor/artifact validation and private factory host |

Generic strings such as `decoder_kv_cache`, `decoder_moe`,
`vision_language`, or `encoder_only` are not the current strategy inventory.
Use `src/runtime/models/*/MODEL.toml` for the live keys.

## Native build-time discovery

`cmake/trtmc_pipeline_plugins.cmake` scans
`src/runtime/models/*/MODEL.toml`, validates each descriptor, builds the
declared plugin sources, and generates runtime-manifest metadata. A contributor
adds a runtime by adding its model-owned descriptor and sources, not by
editing a registration table in the factory.

## Optimized build-time selection

The Python builder first resolves the owning family, then probes only that
family's `IMPLEMENTATION.toml` and exact profile TOMLs. A selected adapter
must also have current producer qualification. This delegated path is
family-owned but is separate from the native CMake descriptor inventory.

## Runtime resolution

A native bundle must identify the runtime strategy. The loader resolves the
owning library, loads it, and looks up the strategy in `PipelineRegistry`.
Failure to load the DSO or find the strategy is an error; the runtime does not
silently select an unrelated generic pipeline.

An optimized bundle identifies itself with `optimized_runtime.json` and
embeds its exact `libtrtmc_impl_*.so` plus artifact tree. The optimized host
validates and materializes that tree and calls the implementation's private
factory ABI. It bypasses the native strategy index, model DSO, and backend
DSO, and any failure is terminal rather than a native fallback.

Embedding the implementation DSO does not make an optimized bundle a complete
runtime image. The host supplies the compatible NVIDIA driver, CUDA runtime,
TensorRT, dynamic loader, and system libraries. Native bundles likewise rely
on host runtime dependencies and load their model/backend DSOs from the
installation rather than from the bundle.
