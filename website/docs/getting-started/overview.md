---
title: Getting Started
description: The shortest supported path from a compatible NVIDIA environment to one verified text inference.
---

Getting Started has one goal: run one Qwen request through
TensorRT-Model-Connect. Follow these pages in order.

| Step | Page | You are done when |
| --- | --- | --- |
| 1 | [System Requirements](environment-and-repro.md) | The GPU is visible and you selected wheel or source. |
| 2 | [Installation](installation.md) | The installed CLI reports TensorRT support. |
| 3 | [Build from Source](source-build.md) | Optional: the CLI and native DSOs are built for the selected GPU. |
| 4 | [Quick Start](quick-start.md) | `./qwen3-0.6b.bundle` is built, inspected, and returns generated text. |

The Quick Start uses a bounded cache profile intended for the first portable
inference. A successful run proves the selected environment can resolve the
checkpoint, build a bundle, load the native Qwen runtime, and execute one text
request. It does not qualify every model or hardware profile on that machine.

After the first run, continue to [Learning Path](../learning-path.md).

{/* Collaborative review anchor: batch 2. */}
