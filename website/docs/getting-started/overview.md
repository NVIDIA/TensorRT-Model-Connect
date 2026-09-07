---
title: Getting Started
description: The shortest supported path from a compatible NVIDIA environment to one verified text inference.
---

Getting Started has one goal: build one bundle and execute one native text
request. Follow these pages in order.

| Step | Page | You are done when |
| --- | --- | --- |
| 1 | [System Requirements](environment-and-repro.md) | The GPU is visible and you selected wheel or source. |
| 2 | [Installation](installation.md) | The Python builder and native CLI are available. |
| 3 | [Build from Source](source-build.md) | Source users built the CLI, backend, and selected family DSO. |
| 4 | [Quick Start](quick-start.md) | The bundle builds, `trtmc inspect` identifies it, and native execution returns text. |

A successful run proves only the selected checkpoint, family, bundle, runtime
directory, and GPU environment. It does not qualify every family or hardware
configuration on the machine.

After the first run, read [Inference Fundamentals](inference-fundamentals.md),
choose a task in [User Guides](../user-guides/overview.md), or browse the
[Learning Path](../learning-path.md).
