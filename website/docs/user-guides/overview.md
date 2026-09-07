---
title: User Guides
description: Goal-oriented guides for building, inspecting, running, and validating family-owned bundles.
---

Use these pages while doing real work. Confirm the exact checkpoint and task in
[Models & Recipes](../models-recipes/overview.md) before treating a generic
command as a support claim.

| Goal | Guide | Result |
| --- | --- | --- |
| Create an artifact | [Build a Bundle](build-a-bundle.md) | A format-1 `.bundle` built by exactly one family. |
| Diagnose an artifact | [Inspect a Bundle](inspect-a-bundle.md) | Family, task, backend, and section inventory. |
| Execute a task | [Run Inference](run-inference.md) | Correct Task command and typed JSON/media output. |
| Place a setting correctly | [Configure Runtime Behavior](configure-runtime.md) | Build, load, or request input at its typed boundary. |
| Establish evidence | [Validate & Benchmark](validate-benchmark.md) | Reproducible correctness or performance evidence. |

| Workload | Guide | Common command |
| --- | --- | --- |
| Decoder text, encoding, embedding, reranking | [Text Generation](text-generation.md) | `run`, `encode`, `embed`, `rerank` |
| Vision-language, ASR, TTS, speech sessions | [Multimodal & Speech](multimodal-speech.md) | `run --image`, `transcribe*`, `generate-audio`, `speak`, `speech-session` |
| Diffusion and perception | [Image & Video](image-video-generation.md) | `generate-image`, `generate-video`, `classify`, `segment`, `geometry` |
| Forecasting and neural operators | [Time-Series](time-series.md) | `forecast`, `solve` |

Use [Tutorials](../learning-path.md) for progressive learning and
[Reference](../api/overview.md) for exact API lookup.
