---
title: Runtime Lifecycle
description: How PipelineFactory loads a bundle, constructs a pipeline, and serves requests.
---

import Diagram from '@site/src/components/Diagram';

The C++ runtime begins at `trtmc::load()` or
`PipelineFactory::from_bundle()`. It reads the bundle header before choosing one
of two mutually exclusive construction paths.

## Authoritative pipeline-load sequences

<Diagram
  src="/img/diagrams/architecture/native-bundle-load.svg"
  alt="Native bundle load sequence reading runtime strategy metadata, loading the owning model and compatible backend DSOs, and creating a concrete pipeline"
  caption="After the header does not claim an optimized descriptor, the native route materializes config.json, resolves strategy ownership and plugin registration, then loads a compatible backend and constructs PipelineContext."
  sequence
/>

<Diagram
  src="/img/diagrams/architecture/optimized-bundle-load.svg"
  alt="Optimized bundle load sequence validating the descriptor and embedded artifact tree, loading the exact implementation DSO, and creating a public pipeline"
  caption="Descriptor presence claims the optimized route; identity, integrity, DSO, or private-factory failures are terminal and never fall back to native."
  sequence
/>

The order matters. On the native path, strategy ownership and plugin lookup are
established before backend loading. On the optimized path, descriptor presence
claims the path before native materialization; an invalid descriptor, artifact,
DSO, or factory is terminal.

## Native dispatch

For a native bundle:

1. `config.json` supplies `runtime_strategy` and strategy-specific metadata.
2. Generated manifest data maps that strategy to one model owner and its derived library.
3. `PipelinePluginLoader` loads the owning
   `libtrtmc_model_<owner>.so`.
4. The exported registrar publishes the declared `IPipelinePlugin` into
   `PipelineRegistry`.
5. `BackendLoader` resolves a compatible backend DSO and TensorRT ABI.
6. The factory resolves run-time configuration and creates `PipelineContext`.
7. The model plugin validates sections, creates modules and helpers, and
   returns a concrete `IPipeline`.

There is no legacy strategy-alias normalization. An unknown strategy,
unavailable model DSO, undeclared registration, incompatible backend, or
invalid required section fails explicitly.

The owner manifest lives at
`python/tensorrt_model_connect/models/<owner>/MODEL.toml`. CMake uses it to
generate the strategy-to-library index and model registrar entry points;
contributors do not append a family switch inside `PipelineFactory`.

## Optimized dispatch

For an optimized bundle, `OptimizedRuntimeHost`:

1. reads and validates `optimized_runtime.json`;
2. reads bounded private implementation metadata;
3. verifies and materializes the embedded artifact tree;
4. opens the exact embedded `libtrtmc_impl_*.so`;
5. validates the versioned private factory, implementation identity, and
   toolchain/runtime contract; and
6. asks that factory to create the public `IPipeline`.

The generic host treats model-owned implementation metadata as opaque. It does
not substitute an installed same-name DSO, use the native strategy index, or
select a native backend DSO.

Embedding the implementation library does not make the bundle hermetic. The
host still supplies compatible driver, CUDA, TensorRT, loader, and system
libraries.

## Plugin construction

`IPipelinePlugin::create()` receives a `PipelineContext` containing:

- the materialized bundle and parsed base config;
- original config text and bundle path;
- the selected `IBackend`;
- Python helper and runtime-cache paths;
- CUDA-graph and KV-cache load options;
- a resolved `ConfigBundle` when resolution succeeded.

The plugin reads its required sections, creates one or more `ITrtModule`
instances through `IBackend`, constructs model-owned state and preprocessing
helpers, then returns a concrete pipeline.

## Request sequence: text generation

Text generation illustrates the common request boundary without implying that
every modality uses a decoder.

<Diagram
  src="/img/diagrams/architecture/text-generation-request.svg"
  alt="Text-generation request sequence covering tokenization, one prefill, sampling, semantic stop checks before the next engine step, token-budget loop control, KV-cache updates, and TextResult construction"
  caption="Prefill produces the first logits. EOS or an answer-stop skips the next engine step; max_new_tokens is the loop boundary, so the last budgeted non-stop token still runs through the engine and advances the cache before the loop exits."
  sequence
/>

Other pipelines implement only the public task methods they support:

| Task shape | Typical runtime work |
| --- | --- |
| Vision-language | Image preprocessing, vision execution, embedding injection, text generation |
| Speech recognition | Audio preprocessing, encoder/decoder or RNNT state, token decoding |
| Diffusion/image/video | Prompt encoding, denoising schedule, component engines, media decode |
| Encoder/embedding/reranking | Tokenization, encoder execution, pooling or scoring |
| Segmentation/detection | Image preprocessing, model execution, geometric postprocessing |
| Time series/operator | Numeric tensor preparation, engine execution, structured output |

The method's presence in `IPipeline` is not model-support evidence. Default
implementations throw when a concrete pipeline does not support an operation.

## Configuration behavior

For native construction, the factory currently supplies:

- schema defaults;
- an optional `defaults` object from materialized `config.json` as bundle
  defaults; and
- `LoadOptions.config_path` plus `LoadOptions.set_tokens` as the session
  request.

Although `ConfigBundle` defines build-time and platform-profile layer types,
this factory path does not inject separate contributions for them.

Native `PipelineFactory` calls fail closed. Missing or malformed
`config.json`, missing `runtime_strategy`, unknown schema namespaces or fields,
invalid values, and invalid explicit overrides all terminate construction
before the model plugin is created. A successful construction supplies a
non-null resolved config to the plugin.

## Concurrency and pipeline pools

One `IPipeline` owns mutable execution context, CUDA stream, cache/state, and
adapter bindings. The public interface does not promise concurrent calls on one
instance.

For native bundles, `PipelineFactory::from_bundle_pool()` creates independent
lanes and returns a `PipelinePool`. A move-only lease gives one request
exclusive access to one lane. Optimized bundles are rejected by this native
pool API because the delegated implementation owns batching and scheduling.

## Shape and engine constraints

Run-time inputs must fit the optimization profiles baked into the selected
artifact. Do not assume that two models in the same modality share engine
sections, batch limits, prefill/decode layout, KV capacity, or dynamic-shape
support. Use bundle inspection plus the owning build/profile contract.

## Runtime source map

| Concern | Source |
| --- | --- |
| Public task API | `include/trtmc/pipeline.h` |
| Pipeline factory | `src/runtime/registry/pipeline_factory.cpp` |
| Native DSO loader | `src/runtime/registry/pipeline_plugin_loader.cpp` |
| Native registry | `src/runtime/registry/pipeline_registry.cpp` |
| Optimized host | `src/runtime/providers/optimized_runtime_host.cpp` |
| Plugin context | `include/trtmc/runtime/pipeline_plugin.h` |
| Backend abstraction | `include/trtmc/runtime/trt_backend.h`, `include/trtmc/runtime/trt_module.h` |
| Model pipelines | `python/tensorrt_model_connect/models/<owner>/runtime/` |

{/* Collaborative review anchor: batch 2. */}
