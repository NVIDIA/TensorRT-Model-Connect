---
title: Tutorial Curriculum
description: A course-style path with progressive modules, hands-on checkpoints, and self-assessment.
---

import Diagram from '@site/src/components/Diagram';

Tutorials teach; they are not a lookup reference. Each module introduces one
mental model, asks you to change or inspect something concrete, and ends with a
self-check. Start after completing [Getting Started](getting-started/overview.md);
the first modules reuse the Qwen bundle from the Quick Start instead of asking
you to rebuild it.

Use [User Guides](user-guides/overview.md) when you already know the concept
and need a command for speech, diffusion, text, or another task. Use
[Reference](api/overview.md) for exact option and API details.

## Course contract

Each module has four parts, adapted from strong public technical courses and
hands-on learning sites:

1. **Learning objectives** state what you should be able to explain or do.
2. **Lab** applies one concept to a real bundle, task, or source boundary.
3. **Evidence checkpoint** gives an observable success condition.
4. **Self-check** asks you to predict or explain the behavior before opening
   the answer key.

Do not measure progress by page completion. Move on when you can satisfy the
module milestone and answer its self-check without copying the prose.

<Diagram
  src="/img/diagrams/learning/course-map.svg"
  alt="Seven-stage learning path numbered zero through six from first inference through advanced validation and optional contribution work"
  caption="Complete the shared text path first, then choose the modality branches that match your workload."
/>

You do not have to complete every modality. Follow the common foundations
through text generation, then choose the task labs that match your workload.

## Module 0: Complete the first inference

Read and run these pages in order:

1. [System Requirements](getting-started/environment-and-repro.md)
2. [Installation](getting-started/installation.md)
3. [Quick Start](getting-started/quick-start.md)

**Milestone:** `trtmc run` returns generated text from
`./qwen3-0.6b.bundle`, and `inspect` reports the `qwen` family and
`qwen_decoder_kv_cache` runtime strategy.

If that milestone does not pass, stay in Getting Started. Architecture,
quantization, and other model recipes will add variables without fixing the
environment boundary.

## Module 1: Learn the bundle workflow

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
and a `.bundle` bundle are different artifacts.

## Module 2: Control text generation

Continue with [Text Generation](tutorials/beginner/text-generation.md). Reuse
`./qwen3-0.6b.bundle` to compare greedy decoding with sampling
controls such as temperature, top-k, top-p, min-p, and a fixed seed.

Use the [CLI Reference](api/cli-reference.md) when you need the exact option
surface. The tutorial teaches the behavior; the API menu is the lookup source.

**Milestone:** you can choose deterministic or sampled decoding intentionally
and can reproduce a seeded request.

## Module 3: Choose another modality

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

## Module 4: Tune advanced behavior

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
and can distinguish build-time topology metadata from launcher/runtime process
count.

## Module 5: Validate and benchmark

Follow [Validation and Benchmarking](tutorials/advanced/validation-and-benchmarking.md),
then use [Testing](reference/testing.md) and
[Benchmarking](reference/benchmarking.md) as references.

Record the exact model revision, bundle configuration, hardware/software
cohort, command, oracle, and thresholds. A successful documentation build or a
single plausible output is not model-parity evidence.

**Milestone:** another developer can reproduce what you measured and can tell
which claims your evidence does and does not support.

## Module 6: Optional architecture and contributor branch

Stop here if you only need to use the product. To understand or change the
implementation, continue with:

1. [Developer Guide](developer-guide/overview.md)
2. [Architecture Overview](architecture/overview.md)
3. [Units and Ownership](architecture/units-and-ownership.md)
4. [Build Pipeline](architecture/build-pipeline.md)
5. [Runtime Lifecycle](architecture/runtime-lifecycle.md)
6. [Validation Design](architecture/validation-design.md)
7. [Extension Overview](extend/overview.md)

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

{/* Collaborative review anchor. */}
