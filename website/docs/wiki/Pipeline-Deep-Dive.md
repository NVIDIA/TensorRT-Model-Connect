# Pipeline Deep Dive

Status: current implementation summary.

## Creation path

1. A CLI, C-linkage C++ shim, or C++ caller provides a `.trtfb` path and
   optional runtime configuration. The C-linkage subset is not a complete
   pure-C ownership API.
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

The public `IPipeline` interface at this revision does not declare
thread-safety. Treat one instance as non-concurrent unless its concrete
implementation documents a stronger guarantee. Native callers that need
independent execution lanes can use `PipelineFactory::from_bundle_pool()`;
that implementation explicitly rejects optimized-runtime bundles.

## Configuration errors

The C++ CLI resolves explicit `--config`/`--set` input before dispatch and
exits nonzero on invalid input. A direct `PipelineFactory` call currently
catches a runtime-config resolution exception, warns, and continues with
a null `runtime_config`; each plugin then chooses its local fallback behavior.
Successful factory resolution attempts to write an effective-config sidecar.
Failure to write that diagnostic file leaves the resolved config active;
failed config resolution does not produce a new sidecar. Library callers that
require fail-fast behavior must treat the resolution warning as an error.

## Inspect the live contract

```bash
./build/trtmc inspect /path/to/model.trtfb
./build/trtmc --help
```

These commands require a built CLI. The bundle passed to `inspect` must be
readable and valid. `inspect` reads bundle metadata without creating a pipeline
or loading a model DSO; execution commands need the owning model DSO to be
discoverable.
