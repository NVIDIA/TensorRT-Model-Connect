---
title: Building Blocks
---

This page maps the abstract building blocks in TensorRT-Model-Connect to the code that implements them.

```mermaid
flowchart LR
  Family["Python family package"] --> Bundle[".trtfb bundle"]
  Bundle --> Loader["generated strategy-to-DSO index"]
  Loader --> ModelDSO["model-owned runtime DSO"]
  ModelDSO --> Pipeline["IPipeline implementation"]
  Backend["TensorRT backend DSO"] --> Pipeline
  Pipeline --> API["C++ API / CLI / C ABI"]
```

Most model additions span the Python family, bundle metadata, model-owned
runtime DSO, and E2E descriptor. Shared infrastructure supplies the contracts
and loading path.

## Beginner Map

Start with these seven blocks before reading the full source-level map:

| Layer | Block | Question it answers |
| --- | --- | --- |
| Source model | HuggingFace checkpoint | What model files did we start from? |
| Build adapter | `FamilyPlugin` | Which Python code understands this model family? |
| Engine artifact | TensorRT engine plan | What optimized GPU execution bytes did we build? |
| Bundle contract | `.trtfb` | What exactly crosses from Python build time to C++ run time? |
| Runtime dispatch | `runtime_strategy` | Which model-owned DSO and plugin should load this bundle? |
| Runtime construction | `IPipelinePlugin` | Which plugin creates the concrete pipeline? |
| User API | `IPipeline` | Which task method does the application call? |

The full map below expands those seven blocks into the concrete helper units used by builders, bundles, plugins, backends, tensors, and tests.

## Ownership Layers

| Layer | Owns | Typical edit |
| --- | --- | --- |
| Beginner/core path | `FamilyPlugin`, `.trtfb`, model-owned `runtime_strategy`, `IPipeline` | Trace or debug one supported model. |
| Model extension path | Python family package, C++ model DSO, config schema, and E2E descriptor | Add a supported model or model-owned behavior. |
| Infrastructure path | Backend DSOs, registration generation, CMake targets | Change loading, ABI isolation, or build ownership. |

## End-to-end building-block map

```mermaid
flowchart TB
  subgraph Python["Python build-time abstractions"]
    ModelConfig["ModelConfig"]
    FamilyPlugin["FamilyPlugin"]
    GraphOps["graph_ops / graph_blocks"]
    EngineBuilder["engine builders"]
    Quant["Quantization plan/context"]
    BundleWriter["BundleInfo + BundleSection"]
  end

  subgraph Artifact["Artifact abstractions"]
    Trtfb[".trtfb"]
    Header["BundleInfo header"]
    Section["BundleSection payloads"]
    ConfigJson["config.json"]
  end

  subgraph Cpp["C++ runtime abstractions"]
    API["IPipeline + result types"]
    Factory["PipelineFactory"]
    Loader["PipelinePluginLoader"]
    ModelDSO["model-owned DSO"]
    Registry["PipelineRegistry"]
    Plugin["IPipelinePlugin"]
    Backend["IBackend"]
    Module["ITrtModule"]
    Pipeline["Concrete pipeline"]
    State["model-owned inference state"]
    Sampler["model-owned sampler"]
    Tensor["Tensor / DeviceTensor"]
    ConfigBundle["ConfigBundle"]
  end

  ModelConfig --> FamilyPlugin
  FamilyPlugin --> GraphOps
  FamilyPlugin --> EngineBuilder
  FamilyPlugin --> Quant
  EngineBuilder --> BundleWriter
  Quant --> BundleWriter
  BundleWriter --> Trtfb
  Trtfb --> Header
  Trtfb --> Section
  Trtfb --> ConfigJson
  Trtfb --> Factory
  Factory --> ConfigBundle
  Factory --> Loader
  Loader --> ModelDSO
  ModelDSO --> Registry
  Registry --> Plugin
  Plugin --> Backend
  Backend --> Module
  Plugin --> Pipeline
  Pipeline --> API
  Pipeline --> State
  Pipeline --> Sampler
  Pipeline --> Tensor
  Module --> Tensor
```

## Build-time blocks

| Block | Source | What it abstracts | Why it exists |
| --- | --- | --- | --- |
| `ModelConfig` | `python/tensorrt_model_connect/config.py` | HuggingFace config differences. | Many model repos use different key names for the same concepts. This gives builders one typed view. |
| `FamilyPlugin` | `python/tensorrt_model_connect/families/base.py` | A model-family adapter. | Matching, weight loading, graph construction, modality-specific components, and quantization hooks vary by family. |
| `WeightDict` and checkpoint mapper | `python/tensorrt_model_connect/checkpoint_mapper.py` | Normalized weight names and tensors. | Builders need stable tensor names even when checkpoint layouts differ. |
| Family-owned graph helpers | `python/tensorrt_model_connect/families/<family>/graph_ops.py` and `graph_blocks.py` when present | TensorRT graph operations and reusable blocks for one family. | Helpers stay beside the model code that defines their assumptions; there are no repository-root graph helper modules. |
| Family-owned builders | Files such as `python/tensorrt_model_connect/families/qwen/standard_decoder_builder.py` | Engine construction flows. | Different model shapes need different graph topology and profile handling without a central model switch. |
| Quantization context | `python/tensorrt_model_connect/quantization/` | Calibration, scales, formats, exclusions. | Quantization needs model-aware policy without leaking into every builder. |
| Python `ConfigBundle` mirror | `python/tensorrt_model_connect/runtime_config/` | Schema-controlled build/runtime config merge. | Python writes bundle defaults using the same conceptual layers as C++. |
| `BundleInfo` / `BundleSection` | `python/tensorrt_model_connect/bundle_writer.py` | Bundle metadata and named payloads. | The runtime needs a structured artifact, not a directory of unrelated files. |

## Artifact blocks

| Block | Source | What it abstracts |
| --- | --- | --- |
| `.trtfb` magic/header/sections | `python/tensorrt_model_connect/bundle_writer.py`, `src/bundle/bundle_format.cpp` | A portable container format for build output. |
| `BundleInfo` | `include/trtmc/bundle.h`, internal `src/bundle/bundle_format.h` | Fast metadata inspection without constructing a pipeline. |
| `config.json` section | Written by builder, read in `PipelineFactory` and plugins | Runtime strategy, IO map, backend name, and strategy-specific fields. |
| Engine sections | Bundle sections such as `engine_plan`, `vision_engine_plan`, `denoiser_plan` | Serialized TensorRT execution plans. |
| Asset sections | Tokenizer, preprocessor, kernel, scale, and family metadata files | Data needed for preprocessing, postprocessing, or custom execution. |

## Runtime blocks

| Block | Source | What it abstracts | Who should use it |
| --- | --- | --- | --- |
| `IPipeline` | `include/trtmc/pipeline.h` | User-facing task interface and typed results. | Applications, CLI, C ABI, tests. |
| `LoadOptions` | `include/trtmc/pipeline.h` | Bundle load-time knobs. | Applications and CLI. |
| `PipelineFactory` | `src/runtime/registry/pipeline_factory.cpp` | Bundle-to-pipeline construction. | Public API and C ABI. |
| `PipelinePluginLoader` | `src/runtime/registry/pipeline_plugin_loader.cpp` | Strategy-owner lookup, model DSO loading, and registration validation. | The factory and loader tests. |
| `PipelineRegistry` | `src/runtime/registry/pipeline_registry.cpp` | Strategy-to-plugin lookup. | Factory and registration tests. |
| `IPipelinePlugin` | `include/trtmc/runtime/pipeline_plugin.h` | Strategy-specific pipeline construction. | Runtime plugin implementations. |
| `PipelineContext` | `include/trtmc/runtime/pipeline_plugin.h` | The construction context passed to plugins. | Runtime plugin implementations. |
| `IBackend` | `include/trtmc/runtime/trt_backend.h` | Backend DSO interface. | Plugins creating modules. |
| `ITrtModule` | `include/trtmc/runtime/trt_module.h` | Engine execution and tensor introspection without TensorRT headers. | Pipelines and runtime core. |
| `Tensor` | `include/trtmc/runtime/tensor.h` | CPU-side tensor view. | Pipeline input/output binding. |
| `DeviceTensor` | `include/trtmc/runtime/device_tensor.h` | Owned GPU tensor storage. | Runtime core and pipelines. |
| Model inference state | `src/runtime/models/<owner>/inference_state.h` when present | Per-sequence decode state under an owner-specific interface such as `QwenInferenceState`. | That owner's text, recurrent, or hybrid pipeline. |
| Model sampler | `src/runtime/models/<owner>/sampler.h` when present | Token selection from logits under an owner-specific interface such as `QwenISampler`. | That owner's text-generation pipeline. |
| `ConfigBundle` | `include/trtmc/config/config_bundle.h` | Resolved layered runtime config. | Factory and migrated plugins. |

## How the blocks interact during text generation

```mermaid
sequenceDiagram
  participant App
  participant API as IPipeline
  participant Pipe as QwenTextGenerationPipeline
  participant State as QwenInferenceState
  participant Module as ITrtModule
  participant Sampler as QwenISampler
  participant Tensor as Tensor/DeviceTensor

  App->>API: generate(prompt, cfg)
  API->>Pipe: virtual dispatch
  Pipe->>Tensor: create token, mask, position tensors
  Pipe->>State: reset, bind_to(module)
  Pipe->>Module: forward prefill/decode
  Module-->>Tensor: logits / present tensors
  Pipe->>Sampler: sample(logits, cfg)
  Sampler-->>Pipe: next token
  Pipe->>State: advance
  Pipe-->>App: TextResult
```

## Choosing the right extension point

```mermaid
flowchart TD
  Start["What are you changing?"] --> NewModel{"New supported model?"}
  NewModel -- yes --> Capsule["Add Python family + unique strategy + model DSO + E2E descriptor"]
  NewModel -- no --> Runtime{"New runtime behavior for an existing owner?"}
  Runtime -- yes --> Owner["Extend that src/runtime/models/owner directory and MODEL.toml"]
  Runtime -- no --> Public{"New user-visible task contract?"}
  Public -- yes --> API["Extend IPipeline + CLI and C ABI as appropriate"]
  Public -- no --> Config["Use the owning shared or model config schema"]
  Capsule --> Tests["Builder + C++ + exact-model E2E evidence"]
  Owner --> Tests
  API --> Tests
  Config --> Tests
```

Do not point a new family at another model's runtime strategy. Similar models
may reuse source patterns, but each supported model owns a unique strategy key,
DSO, and registration manifest. E2E `task_strategy` is where models with the
same user-visible task are grouped.
