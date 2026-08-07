---
title: User Guides
description: Goal-oriented guides for building bundles, running tasks, configuring behavior, and validating results.
---

User Guides are for quick lookup while doing real work. Each page starts from
a goal, shows the relevant commands and configuration boundaries, and links to
the exact reference surface.

Use [Tutorials](../learning-path.md) instead when you want a course that builds
understanding step by step. Use [Reference](../api/overview.md) when you already
know the concept and need every option or API field.

## Core workflow

| Goal | Guide | Result |
| --- | --- | --- |
| Create an artifact | [Build a Bundle](build-a-bundle.md) | A named `.trtfb` with recorded model/config inputs. |
| Diagnose an artifact | [Inspect a Bundle](inspect-a-bundle.md) | Bundle kind, family, runtime identity, and section inventory. |
| Execute a task | [Run Inference](run-inference.md) | The correct CLI command and typed result for the bundle. |
| Change runtime behavior | [Configure Runtime Behavior](configure-runtime.md) | A validated config file or `--set` override at the right lifecycle layer. |
| Establish evidence | [Validate & Benchmark](validate-benchmark.md) | Reproducible parity, quality, or performance evidence. |

## Task lookup

| Workload | Guide | Common command |
| --- | --- | --- |
| Decoder text, encoder NLP, embedding, reranking | [Text Generation](text-generation.md) | `run`, `encode`, `embed`, `rerank` |
| Vision-language, ASR, TTS, speech-to-speech | [Multimodal & Speech](multimodal-speech.md) | `run --image`, `transcribe`, `generate-audio`, `speak` |
| Diffusion, classification, segmentation | [Image & Video Generation](image-video-generation.md) | `generate-video`, `classify`, `segment`, `segment-prompted` |
| Forecasting and neural operators | [Time-Series](time-series.md) | `solve` |

The supported-model inventory is separate from these instructions. Confirm an
exact checkpoint/configuration in [Models & Recipes](../models-recipes/overview.md)
before treating a generic command as a support claim.
