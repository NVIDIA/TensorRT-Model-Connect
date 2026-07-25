# Pipeline Deep Dive

Status: current implementation summary.

## Creation path

1. A CLI, C ABI, or C++ caller provides a `.trtfb` path and optional runtime
   configuration.
2. `PipelineFactory` reads the bundle header. If `optimized_runtime.json` is
   present, the optimized host validates the descriptor and artifact hash,
   materializes the embedded artifact tree, loads the exact implementation
   DSO, validates its factory/identity, and returns its `IPipeline`. No failure
   on this path falls back to native dispatch.
3. Otherwise the factory reads `config.json` and extracts
   `runtime_strategy`. If absent, it accepts only a generated
   manifest default; otherwise creation fails.
4. Legacy aliases may be normalized using generated compatibility metadata.
5. The plugin loader maps the resolved strategy to a model library and loads
   that DSO.
6. `PipelineRegistry` looks up the registered `IPipelinePlugin`.
7. Runtime schemas and supplied overrides are resolved.
8. The plugin validates the bundle and creates its family-owned `IPipeline`.

The implementation is in:

- `src/runtime/registry/pipeline_factory.cpp`
- `src/runtime/registry/pipeline_plugin_loader.cpp`
- `src/runtime/registry/pipeline_registry.cpp`
- `src/runtime/providers/optimized_runtime_host.cpp`
- `include/trtmc/runtime/pipeline_plugin.h`

## Registration path

For native bundles, each `src/runtime/models/<family>/MODEL.toml` declares a library, registration
source/symbol pairs, and strategy keys. CMake scans those descriptors and
generates the strategy-to-library index. Loading the library invokes its
registration symbol; the plugin then registers the declared keys.

There is no supported contributor step that appends a model to a central
strategy switch or a static list in `trtmc_pipeline_plugins.cmake`.

Optimized bundles do not use that registration path. Their
`optimized_runtime.json` descriptor and embedded `libtrtmc_impl_*.so` are
artifact-bound, and the generic host uses a private versioned factory contract
to obtain the public pipeline.

## Requests

`include/trtmc/pipeline.h` defines the public `IPipeline` operations and typed
result structures. A concrete family pipeline implements only supported
operations; unsupported calls fail explicitly. Text generation, encoding,
embedding, reranking, segmentation, detection, audio, speech, diffusion,
time-series, and other tasks may have different request/state lifecycles even
when they share the same public interface.

A single pipeline instance owns mutable execution context, stream, cache/state,
and adapter bindings; do not issue concurrent requests to one instance. Native
callers can use `PipelineFactory::from_bundle_pool()` to acquire exclusive
leases over independent lanes. Optimized bundles are rejected by that API
because their delegated runtime owns batching and scheduling.

## Configuration errors

The C++ CLI resolves explicit `--config`/`--set` input before dispatch and
exits nonzero on invalid input. A direct `PipelineFactory` call currently
catches a runtime-config resolution exception, warns, and continues with
a null `runtime_config`; each plugin then chooses its local fallback behavior.
Successful resolution writes an effective-config file; failed factory
resolution does not write a new one. Library callers that require fail-fast
behavior must treat the warning as an error.

## Inspect the live contract

```bash
./build/trtmc inspect /path/to/model.trtfb
./build/trtmc --help
```

These commands require a built CLI. The bundle passed to `inspect` must be
readable and valid. `inspect` reads bundle metadata without creating a pipeline
or loading a model DSO; execution commands need the owning model DSO to be
discoverable.
