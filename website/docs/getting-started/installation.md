---
title: Installation
---

Choose one path:

| Path | Current boundary |
| --- | --- |
| Release wheel | Linux aarch64, Python 3.10 or 3.12, glibc 2.39 or newer, and official TensorRT 11.1.0.106. |
| Source build | Linux x86_64 or aarch64 with Docker and NVIDIA Container Toolkit. |

Release wheels currently support aarch64 only. x86_64 wheels are planned; use
[Build from Source](source-build.md) on x86_64 today.

## Install a release wheel

This first-time path uses Python 3.12. Download the aarch64 `+trt111` `py312`
`.whl` asset from the [latest GitHub release](https://github.com/NVIDIA/TensorRT-Model-Connect/releases/latest).
If no matching wheel is published, use the source path instead.

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate

WHEEL=/path/to/downloaded-wheel.whl
python -m pip install "$WHEEL"
```

Python 3.10 users can instead use the `+trt111` `py310` wheel and corresponding
`python3.10` commands. Keep this virtual environment active and continue to
[Quick Start](quick-start.md), where the CLI is checked once.

## Build and run from source

Continue in the source container prepared by
[Build from Source](source-build.md). That page adds the source-built CLI to
`PATH` before sending you to Quick Start.

Do not mix a wheel, backend DSO, bundle, or TensorRT library from another
architecture or TensorRT cohort.

{/* Collaborative review anchor. */}
