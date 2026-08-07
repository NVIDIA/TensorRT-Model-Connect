---
title: Project Overview
description: What TensorRT-Model-Connect provides, who it serves, and where it fits in the TensorRT ecosystem.
---

TensorRT-Model-Connect (TRTMC) provides a common framework and a growing set of
family-owned reference implementations for running diverse model families on
TensorRT. Each implementation shows how to turn a supported Hugging Face or
local checkpoint into a `.bundle` artifact and invoke it through task-oriented
C++ APIs.

The implementations support straightforward deployment, but they are also
intended as blueprints that developers can inspect, modify, extend, and
customize. Model behavior remains visible in family-owned builders, native
runtime pipelines, helper kernels, configuration schemas, and validation
contracts instead of being hidden behind a single generic integration.

## The build and runtime boundary

Python owns checkpoint resolution and TensorRT engine construction at build
time. Native profiles execute model inference in C++ without PyTorch. A small
number of hybrid profiles explicitly invoke a helper Python executable; their
E2E manifests declare that runtime dependency.

The `.bundle` artifact is the handoff between those environments:

```text
Hugging Face or local checkpoint
  -> Python family resolution and TensorRT build
  -> .bundle artifact
  -> native C++ task API
```

Native bundles resolve their matching model and TensorRT backend DSOs at
runtime. Exactly qualified optimized-runtime bundles can carry their own
implementation DSO. Both forms still require a compatible NVIDIA driver,
CUDA/TensorRT cohort, dynamic loader, and system libraries.

There is no intermediate ONNX export step. Applications load a bundle and call
task APIs such as `generate()`, `transcribe()`, `generate_image()`, `embed()`,
or `solve()` instead of maintaining conversion stages and model-specific
application glue.

## Where it fits in the TensorRT ecosystem

Choose the import path that matches the model boundary you already own. Each
path targets TensorRT execution, but the development and deployment interfaces
are different.

| Starting point | Interface | When to use it |
| --- | --- | --- |
| Hugging Face or local checkpoint | **TensorRT-Model-Connect** | Start from a model-family reference implementation, build a `.bundle` for native C++ task inference, and customize the implementation as needed. |
| PyTorch model | **Torch-TensorRT** | Keep the model in the PyTorch ecosystem while compiling its execution with TensorRT. |
| Portable framework interchange | **ONNX** | Use an exchange format when portability across originating frameworks is the primary requirement. |

TRTMC may not be the right boundary when inference already lives entirely in a
Python/PyTorch deployment, or when ONNX is a required interchange artifact.

## Who it is for

TRTMC is designed for teams that:

- want a working TensorRT reference implementation for a supported model
  instead of starting its builder and runtime integration from scratch;
- need inference in a C++ service, embedded application, robotics stack, or
  edge system and want a concrete deployment blueprint to adapt;
- want to study or customize model-specific builders, native runtime pipelines,
  helper kernels, and integration boundaries;
- want one versioned bundle boundary between a Python-first build environment
  and a native application; or
- need a common task API across text, vision, audio, diffusion, segmentation,
  time-series, and other model families.

## What it simplifies

A conventional model-to-deployment path can accumulate several conversion and
integration boundaries:

```text
PyTorch -> ONNX or TorchScript -> TensorRT -> model-specific C++ integration
```

TRTMC reduces that path to a family-owned build and a task-oriented runtime:

| Traditional pain point | TRTMC boundary |
| --- | --- |
| ONNX export failures and unsupported conversion gaps | Family-owned builders compile supported checkpoints directly with TensorRT APIs. |
| Repeated model-specific application integration | Applications load a bundle and use a task-oriented runtime API. |
| Validation across several conversion artifacts | Build, runtime, and E2E manifests identify one bundle contract and its evidence. |
| Python framework dependencies in native inference paths | Native profiles execute model inference in C++; manifests explicitly flag hybrid profiles that require helper Python. |
| Opaque deployment artifacts | `trtmc inspect` exposes bundle kind, model family, precision, runtime identity, and engines. |

## Model coverage and ownership

The build-and-run design spans decoder and hybrid language models,
encoder/embedding/reranking models, translation, vision-language and OCR,
speech recognition and synthesis, diffusion image and video generation,
segmentation, time-series forecasting, and neural operators.

Each model-family implementation keeps its knowledge in family-owned builder,
runtime, and E2E descriptors. The repository also includes agent instructions
and model-local validation contracts so contributors can extend one family
without editing a hand-written global registry.

Declared inventory is not proof that every model passed on every platform. Use
[Supported Models](../models-recipes/overview.md) for exact checkpoint IDs,
architectures, configurations, and evidence levels.

## Trust boundary

TensorRT-Model-Connect is a reference implementation. Users are responsible
for trusting the checkpoints, bundles, native libraries, and local environment
they provide when building or running models. Continue with
[System Requirements](environment-and-repro.md) before selecting an
installation path.

{/* Collaborative review anchor. */}
