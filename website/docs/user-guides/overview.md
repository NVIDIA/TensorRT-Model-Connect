---
title: User Guides
description: Goal-oriented guides for building bundles, running tasks, configuring behavior, and validating results.
---

User Guides are quick references for real work. Tutorials explain the same
ideas progressively; the API pages list the exact public contracts.

## Core workflow

| Goal | Guide | Result |
| --- | --- | --- |
| Create an artifact | [Build a Bundle](build-a-bundle.md) | A family-owned `.bundle` built from an exact checkpoint. |
| Identify an artifact | [Inspect a Bundle](inspect-a-bundle.md) | `format`, `family`, `task`, `backend`, and section offsets. |
| Execute a task | [Run Inference](run-inference.md) | The native command matching the abstract Task interface. |
| Change runtime behavior | [Configure Runtime Behavior](configure-runtime.md) | A setting placed at build, load, or request time. |
| Establish evidence | [Validate & Benchmark](validate-benchmark.md) | Reproducible correctness or performance evidence. |

## Task lookup

| Workload | Guide | Common command |
| --- | --- | --- |
| Decoder text, encoder NLP, embedding, reranking | [Text Generation](text-generation.md) | `run`, `encode`, `embed`, `rerank` |
| Vision-language, ASR, TTS, speech-to-speech | [Multimodal & Speech](multimodal-speech.md) | `run --image`, `transcribe`, `generate-audio`, `speak` |
| Diffusion, classification, segmentation, geometry | [Image & Video Generation](image-video-generation.md) | `generate-image`, `generate-video`, `classify`, `segment`, `geometry` |
| Forecasting and neural operators | [Time-Series](time-series.md) | `forecast`, `solve` |

Confirm an exact checkpoint in
[Supported Models](../models-recipes/overview.md). A generic command existing
does not mean that every family implements its Task interface.

Examples and benchmark applications consume these same public APIs in one
direction. Their code is not imported by core or a family.
