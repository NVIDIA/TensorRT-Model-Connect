---
title: Learning Path
description: A course-style path for learning TensorRT-Model-Connect from inference fundamentals to extension work.
---

import Diagram from '@site/src/components/Diagram';

This page is the tutorial index and the recommended order for learning the
project. Start only after completing [Getting Started](getting-started/overview.md);
the first stages reuse the Qwen bundle from the Quick Start instead of asking
you to rebuild it.

<Diagram
  src="/img/diagrams/learning/course-map.svg"
  alt="Seven-stage learning path numbered zero through six from first inference through advanced validation and optional contribution work"
  caption="Complete the shared text path first, then choose the modality branches that match your workload."
/>

You do not have to complete every modality. Follow the common path through text
generation, then choose the branches that match your workload.

Keep one CLI selector in the same shell for the Learning path:

```bash
export TRTMC=trtmc
# Source build inside the development container:
# export TRTMC=./build/trtmc
```

## Stage 0: Complete the first inference

Read and run these pages in order:

1. [Prerequisites and Environment](getting-started/environment-and-repro.md)
2. [Installation](getting-started/installation.md)
3. [Quick Start](getting-started/quick-start.md)

**Milestone:** `trtmc run` (or `./build/trtmc run`) returns generated text from
`Qwen3-0.6B.trtfb`, and `inspect` reports the `qwen` family and
`qwen_decoder_kv_cache` runtime strategy.

If that milestone does not pass, stay in Getting Started. Architecture,
quantization, and other model recipes will add variables without fixing the
environment boundary.

## Stage 1: Learn the bundle workflow

Read:

- [Glossary](getting-started/glossary.md) when a term is unfamiliar.
- [Inference Fundamentals](getting-started/inference-fundamentals.md) for the
  checkpoint-to-result mental model.
- [Inspect Bundles](tutorials/beginner/inspect-bundles.md) for artifact-first
  debugging.

Use the bundle you already built. Inspect its metadata and engine list, then
identify which evidence came from model conversion, which was stored in the
bundle, and which was produced only when the C++ runtime loaded it.

**Milestone:** you can explain why a Hugging Face checkpoint, a TensorRT engine,
and a `.trtfb` bundle are different artifacts.

## Stage 2: Control text generation

Continue with [Text Generation](tutorials/beginner/text-generation.md). Reuse
`Qwen3-0.6B.trtfb` to compare deterministic greedy decoding with sampling
controls such as temperature, top-k, top-p, min-p, and a fixed seed.

Use the [CLI Reference](api/cli-reference.md) when you need the exact option
surface. The tutorial teaches the behavior; the API menu is the lookup source.

**Milestone:** you can choose deterministic or sampled decoding intentionally
and can reproduce a seeded request.

## Stage 3: Choose another modality

Choose one or more branches:

| Goal | Tutorial | What changes from text generation |
| --- | --- | --- |
| Image-conditioned text or speech/audio | [Multimodal and Speech](tutorials/intermediate/multimodal-and-speech.md) | Preprocessing, bundle sections, task method, and output type. |
| Canary ASR decoding details | [Canary Decoding](tutorials/intermediate/canary-decoding.md) | Timestamp-aware token handling and speech-specific decoding. |
| Image/video diffusion or time-series | [Diffusion, Vision, and Time-Series](tutorials/intermediate/diffusion-and-time-series.md) | Denoising or forecasting replaces token-by-token generation. |

The diffusion tutorial includes progressively larger FLUX, PixArt, Wan, and
Thor-qualified Wan2.2 recipes. Do not use the Thor recipe as a first-run
environment test.

Use [Model Recipes](getting-started/build-and-run.md) only as an optional task
index. Each recipe can add model-specific dependencies and hardware demands;
it is not another Getting Started path.

**Milestone:** for the modality you chose, you can name the input
preprocessing, engine components, public task method, and returned result type.

## Stage 4: Tune advanced behavior

Read:

- [Quantization and Runtime Knobs](tutorials/advanced/quantization-and-runtime-knobs.md)
- [Run Inference on Multiple GPUs](tutorials/advanced/multi-device-inference.md)
- [Sampling](features/sampling.md)
- [Quantization](features/quantization.md)
- [Multi-Device Execution](features/multi-device.md)
- [Configuration and Backends](features/config-and-backends.md)

Change one variable at a time and inspect the resulting bundle. A smaller or
faster build is not automatically an accuracy-equivalent build. Multi-device
topologies also require an exact model manifest and one runtime rank per GPU;
the generic build flag alone does not establish support.

**Milestone:** you can identify whether a setting belongs to build-time model
conversion, bundle metadata, runtime configuration, or request-time decoding,
and can distinguish requested world size from a model-owned subgroup topology.

## Stage 5: Validate and benchmark

Follow [Validation and Benchmarking](tutorials/advanced/validation-and-benchmarking.md),
then use [Testing](reference/testing.md) and
[Benchmarking](reference/benchmarking.md) as references.

Record the exact model revision, bundle configuration, hardware/software
cohort, command, oracle, and thresholds. A successful documentation build or a
single plausible output is not model-parity evidence.

**Milestone:** another developer can reproduce what you measured and can tell
which claims your evidence does and does not support.

## Stage 6: Optional architecture and contributor branch

Stop here if you only need to use the product. To understand or change the
implementation, continue with:

1. [Architecture Overview](architecture/overview.md)
2. [Units and Ownership](architecture/units-and-ownership.md)
3. [Build Pipeline](architecture/build-pipeline.md)
4. [Runtime Lifecycle](architecture/runtime-lifecycle.md)
5. [Validation Design](architecture/validation-design.md)
6. [Extension Overview](extend/overview.md)

Then choose the extension guide owned by your change:

- [Add a Model Family](extend/add-model-family.md)
- [Add an Optimized Runtime Implementation](extend/add-optimized-runtime.md)
- [Add a Runtime Strategy](extend/add-runtime-strategy.md)
- [Add a Configuration Schema](extend/add-config-schema.md)
- [Validate a Model Contribution](extend/model-validation.md)
- [Bring Your Own Kernel](tutorials/advanced/bring-your-own-kernel.md)
- [Contributing](extend/contributing.md)

Bring Your Own Kernel is an advanced extension workflow, not part of the
beginner or common user path. Use it only when replacing or adding a kernel
implementation is the goal.

**Milestone:** before editing, you can name the model-owned Python descriptor,
runtime strategy and DSO owner, public API boundary, tests, E2E manifest, and
documentation that form the vertical slice of your change.
