---
slug: /
title: TensorRT-Model-Connect Documentation
description: User documentation for building TensorRT bundles from HuggingFace models and running them from the C++ runtime.
---

import useBaseUrl from '@docusaurus/useBaseUrl';


TensorRT-Model-Connect is a deployment stack for deep learning inference.

It has one job: take Python-first AI checkpoints, compile the performance-critical pieces into TensorRT artifacts, and run them through a native C++ task API.

If any phrase in that sentence is new, use this short translation:

| Phrase | Meaning |
| --- | --- |
| Deep learning inference | Running a trained model on a new request. |
| Python-first checkpoint | Model files released for Python libraries such as HuggingFace Transformers or Diffusers. |
| TensorRT artifact | A compiled GPU execution plan built for an NVIDIA inference environment. |
| Native C++ task API | A C++ interface with methods such as `generate`, `transcribe`, `generate_image`, `segment`, and `solve`. |

<figure className="trtmc-diagram trtmc-diagram--wide">
  <div className="trtmc-diagram__media">
    <img src={useBaseUrl('/img/diagrams/trtmc-course-map.svg')} alt="Course-style learning map for TensorRT-Model-Connect" />
  </div>
  <figcaption>Start with concepts, prove them with one working bundle, then move into architecture and extension work.</figcaption>
</figure>

If you are new to inference, start with this mental model:

- A trained model is a function with learned weights.
- Inference means running that function on new input.
- TensorRT turns the model's math graph into an optimized GPU engine.
- TensorRT-Model-Connect either builds native TensorRT engines or invokes an
  exact qualified family-owned optimized-runtime provider, then packages the
  result into a `.trtfb` bundle.
- The C++ runtime loads a bundle and exposes task methods such as `generate`, `transcribe`, `generate_image`, `segment`, and `solve`.

```mermaid
flowchart LR
  UserInput["User input<br/>text, image, audio, time-series"] --> Pipeline["C++ IPipeline method"]
  Pipeline --> Preprocess["Preprocess<br/>tokens, pixels, mel features"]
  Preprocess --> Engine["TensorRT engine<br/>GPU execution"]
  Engine --> Postprocess["Postprocess<br/>sample token, decode text, write image"]
  Postprocess --> Result["Typed result<br/>TextResult, ImageResult, AudioResult"]
```

The project is intentionally split into two phases:

- Python resolves the checkpoint and family. A qualified provider profile can
  build a delegated optimized artifact; otherwise a native family plugin
  builds TensorRT plans.
- C++ loads either `.trtfb` shape and returns a task-specific `IPipeline`.
- The bundle is the contract between build and runtime. Native bundles carry
  plans, assets, model metadata, and `runtime_strategy`; optimized bundles
  carry `optimized_runtime.json` and an embedded implementation DSO/artifact
  tree.

```mermaid
flowchart TB
  subgraph Build["Build phase: Python"]
    HF["HuggingFace model directory"] --> Config["ModelConfig"]
    Config --> Route{"qualified provider<br/>profile matches?"}
    Route -->|no| Family["native FamilyPlugin"]
    Family --> TRT["TensorRT engine plans"]
    Route -->|yes| Provider["family provider adapter"]
    Provider --> Bundle[".trtfb bundle"]
    TRT --> Bundle
  end

  subgraph Run["Run phase: C++"]
    Bundle --> Factory["PipelineFactory"]
    Factory --> Kind{"optimized_runtime.json?"}
    Kind -->|no| Loader["model-plugin loader"]
    Loader --> DSO["owning libtrtmc_model_*.so"]
    DSO --> Plugin["IPipelinePlugin"]
    Kind -->|yes| Optimized["embedded libtrtmc_impl_*.so"]
    Plugin --> Runtime["Concrete IPipeline"]
    Optimized --> Runtime
    Runtime --> Output["Task output"]
  end
```

This site is organized for users first:

<div class="trtmc-card-grid">
  <div class="trtmc-card">
    <strong>New to inference</strong>
    Learn tensors, tokens, logits, engines, prefill, decode, KV cache, and bundles before touching internals.
  </div>
  <div class="trtmc-card">
    <strong>Trying the project</strong>
    Build one bundle, inspect the artifact, and run deterministic text generation from the C++ runtime.
  </div>
  <div class="trtmc-card">
    <strong>Reading the architecture</strong>
    Follow both source-level paths: native family plugin to runtime strategy,
    and qualified optimized adapter to embedded implementation and concrete
    pipeline.
  </div>
  <div class="trtmc-card">
    <strong>Extending support</strong>
    Decide whether your change belongs in a family plugin, runtime plugin, config schema, pipeline, or test.
  </div>
</div>

| Start here | Use this when |
| --- | --- |
| [Learning Path](learning-path.md) | You want the course-style sequence from zero background to extension work. |
| [Glossary](getting-started/glossary.md) | You want plain definitions for inference, deployment, and project terms. |
| [Environment and First Repro](getting-started/environment-and-repro.md) | You want to prove the source-build container, Python builder, and C++ runtime work before building a model. |
| [Inference Fundamentals](/getting-started/inference-fundamentals) | You are new to deep learning inference or TensorRT deployment. |
| [Quick Start](getting-started/quick-start.md) | You want the shortest path from install or checkout to a generated answer. |
| [Tutorials](tutorials/beginner/text-generation.md) | You want a guided path from beginner to advanced usage. |
| [API Manual](api/overview.md) | You need exact CLI, Python, or C++ entry points. |
| [Architecture](architecture/overview.md) | You need to understand how builder, bundle, backend, and pipeline pieces fit. |
| [Unit Design](unit-design/overview.md) | You are modifying internals and need source-level responsibilities. |
| [Extend](extend/overview.md) | You are adding a model family, runtime strategy, or config namespace. |

## What this project is doing

Most modern AI models are released as Python-first artifacts: a `config.json`, tokenizer files, model weights, and Python model code or library classes. That format is excellent for research and experimentation, but production inference often needs different properties:

- Native runtime integration from C or C++.
- Predictable GPU execution.
- A deployable artifact that does not require the full original Python model stack at request time.
- Clear compatibility with CUDA, TensorRT, GPU architecture, quantization format, and runtime settings.
- A task-oriented API across the text, audio, image, video,
  vision-language, segmentation, classification, translation, and time-series
  models represented by the current descriptors. A detection method/command is
  reserved in the API, but no current model descriptor or E2E manifest claims
  object-detection support.

TensorRT-Model-Connect addresses that by separating "understand the model" from "serve the model":

| Concern | Where it lives | Why |
| --- | --- | --- |
| Read HuggingFace config and weights | Python builder | Python has the richest ecosystem for model formats and checkpoint conversion. |
| Construct TensorRT graphs | Python builder | Build-time logic can use TensorRT Python APIs and model-specific adapters. |
| Package engines, tokenizer assets, config, and metadata | `.trtfb` bundle | The bundle becomes the stable build/runtime handoff. |
| Load, validate, and dispatch the bundle | C++ runtime | Deployment code can stay native and task-oriented. |
| Execute engine plans and own request state | C++ pipeline and backend DSO | Request-time latency stays in native code and TensorRT ABI is isolated. |

The model-owned inventories at this revision contain:

- 78 Python family manifests under
  `python/tensorrt_model_connect/families/*/MODEL.toml`.
- 78 E2E family indexes and 203 declared E2E manifests under
  `tests/e2e/models/`.
- 78 C++ runtime model manifests and 79 unique model-owned runtime strategy
  keys under `src/runtime/models/`. Eagle VLM intentionally owns two strategy
  keys.

Do not update these numbers by hand after adding a model. Run
`python3 tools/model_ci.py validate` from the repository root; it validates the
three ownership roots and prints the current model inventory.

## Reading order

New users should read in this order:

1. [Learning Path](learning-path.md) to see the whole course.
2. [Glossary](getting-started/glossary.md) to learn plain-language terms.
3. [Inference Fundamentals](/getting-started/inference-fundamentals) to connect the terms into a model.
4. [Environment and First Repro](getting-started/environment-and-repro.md) to prove your setup.
5. [Quick Start](getting-started/quick-start.md) to run one model.
6. [Inspect Bundles](tutorials/beginner/inspect-bundles.md) before debugging runtime behavior.
7. [Beginner Text Generation](tutorials/beginner/text-generation.md) to understand the full path from prompt to generated token.
8. [Architecture Overview](architecture/overview.md) to connect the tutorial to the actual source tree.
9. [Building Blocks](/unit-design/building-blocks) when you are ready to modify or extend the code.
