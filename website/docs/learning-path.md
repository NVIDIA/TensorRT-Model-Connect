---
title: Tutorial Curriculum
description: A progressive path from first bundle to family-owned contribution.
---

import Diagram from '@site/src/components/Diagram';

Tutorials teach one mental model at a time. Use [User Guides](user-guides/overview.md)
when you already know the concept and need a task recipe, and use
[Reference](api/overview.md) for exact options and APIs.

<Diagram
  src="/img/diagrams/learning/course-map.svg"
  alt="Progressive learning path from first inference through task tutorials, validation, benchmarking, BYOK, and family-owned contribution"
  caption="Complete the common bundle and Task API path first, then choose only the task or contributor branches you need."
/>

## How to use the course

Each stage has an observable milestone. Move on when you can reproduce that
milestone and explain what the evidence proves; page completion alone is not a
result.

## 0. Complete one inference

Read and run:

1. [System Requirements](getting-started/environment-and-repro.md)
2. [Installation](getting-started/installation.md)
3. [Quick Start](getting-started/quick-start.md)

**Milestone:** the build command creates `gpt2.bundle`, `trtmc inspect` reports
`family: gpt2`, `task: text_generation`, and `backend: trt`, and `trtmc run`
returns text through the loaded family Task implementation.

## 1. Understand the artifact boundary

Read [Inference Fundamentals](getting-started/inference-fundamentals.md) and
[Inspect Bundles](tutorials/beginner/inspect-bundles.md). Identify what belongs
to the checkpoint, the TensorRT plan, the bundle container, the family DSO, and
the backend DSO.

**Milestone:** you can explain why a readable bundle is not proof that its
engine loaded or its output is correct.

## 2. Control a request

Follow [Text Generation](tutorials/beginner/text-generation.md). Compare greedy
decoding with a seeded sampled request, changing one option at a time. Use the
[CLI Reference](api/cli-reference.md) as the option lookup source.

**Milestone:** you can reproduce the same seeded request and distinguish
request-time sampling from build-time precision and shape settings.

## 3. Choose another Task interface

Choose only the branch relevant to your workload:

| Goal | Tutorial |
| --- | --- |
| Vision-language, speech recognition, or audio generation | [Multimodal and Speech](tutorials/intermediate/multimodal-and-speech.md) |
| Canary decoding behavior | [Canary Decoding](tutorials/intermediate/canary-decoding.md) |
| Diffusion, segmentation, or time series | [Diffusion, Vision, and Time-Series](tutorials/intermediate/diffusion-and-time-series.md) |

**Milestone:** you can name the concrete input preparation, family-owned engine
sequence, abstract Task method, and returned result type for your chosen case.

## 4. Change build or runtime behavior deliberately

Continue with:

- [Quantization and Runtime Knobs](tutorials/advanced/quantization-and-runtime-knobs.md)
- [Multi-Device Inference](tutorials/advanced/multi-device-inference.md)
- [Bring Your Own Kernel](tutorials/advanced/bring-your-own-kernel.md)

Every option is family-specific unless the public contract says otherwise.
Generic TP/CP or quantization arguments do not prove that every family supports
them. A collective-capable family owns its communicator and NCCL dependency.

**Milestone:** you can locate the setting's owner and its negative validation,
and you do not infer support from parser acceptance alone.

## 5. Validate and benchmark

Follow [Validation and Benchmarking](tutorials/advanced/validation-and-benchmarking.md),
then use [Testing](reference/testing.md), [Benchmarking](reference/benchmarking.md),
and [Profiling](reference/profiling.md) as references.

**Milestone:** another developer can reproduce the exact revision, model,
bundle settings, hardware/software cohort, input, oracle, timing boundary, and
result. Your report clearly separates model correctness from performance.

## 6. Optional contributor path

Before editing a family, read:

1. [Developer Guide](developer-guide/overview.md)
2. [AI-Native Horizontal Scaling Architecture](architecture/ai-native-horizontal-scaling.md)
3. [Add a Model Family](extend/add-model-family.md)
4. [Model Validation](extend/model-validation.md)
5. [Contributing](extend/contributing.md)

**Milestone:** you can describe the intended change as a diff under
`families/<family>/**` and name the exact family-owned E2E that closes the
checkpoint-to-Task loop. If the change requires shared code, you can explain
the existing public contract that makes that shared edit necessary.
