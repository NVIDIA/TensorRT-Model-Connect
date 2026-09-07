---
title: Project Overview
description: What TensorRT-Model-Connect provides, who it serves, and where it fits in the TensorRT ecosystem.
---

TensorRT-Model-Connect (TRTMC) is a collection of self-contained TensorRT model
reference implementations. Each family shows how to turn a supported Hugging
Face or local checkpoint into a `.bundle` and expose it through an abstract C++
Task API.

The implementations are deployment paths and readable blueprints. A family
keeps its checkpoint matching, TensorRT graph, weights, bundle sections, native
pipeline, preprocessing, postprocessing, dependencies, and validation together
under `families/<family>/`.

## The build and runtime boundary

Python resolves the checkpoint and builds TensorRT engines. Native C++ loads
the resulting bundle and executes the family implementation:

```text
Hugging Face ID or local checkpoint
  -> family/support.py selects exactly one owner
  -> that family's plain model.py builds TensorRT sections
  -> .bundle
  -> trtmc loads the named backend and family DSO from --runtime-root
  -> the family returns an abstract Task interface
  -> the application calls generate(), transcribe(), segment(), or another task
```

There is no intermediate ONNX export. The shared core owns only model
resolution, bounded bundle I/O, stable interfaces, loading, and control
transfer. It does not own model graphs, dispatch policy, or model behavior.

The family builder is a plain `build(request, writer)` function. It must not
inherit from a builder base class. Families must not import, link, or otherwise
depend on one another; copying small amounts of implementation is preferable
to creating a shared model layer.

## One-way application dependencies

User programs, examples, benchmark tools, and bring-your-own-kernel workflows
are applications of the public build, load, Engine, and Task contracts:

```text
apps and examples -> public ModelConnect APIs <- family implementations
```

Core and families never depend on application code. That one-way relationship
keeps an example useful without turning it into required infrastructure.

## Choose the right TensorRT path

TensorRT-Model-Connect is the broad path for exploring and adapting supported
model families. For production LLM/VLM deployment on NVIDIA edge platforms
where deployment performance is the priority, start with
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).

Other starting points serve different needs:

| Starting point | Interface | When to use it |
| --- | --- | --- |
| Hugging Face or local checkpoint | **TensorRT-Model-Connect** | Use a family-owned implementation, build a bundle, and embed a native Task API. |
| Production edge LLM/VLM | **TensorRT Edge-LLM** | Start here when its supported deployment surface and performance are the goal. |
| PyTorch model | **Torch-TensorRT** | Keep the application in PyTorch while compiling execution with TensorRT. |
| Portable framework interchange | **ONNX** | Use an exchange format when framework portability is required. |

## Who it is for

TRTMC is designed for teams that:

- want a concrete TensorRT implementation for a supported checkpoint;
- need native inference in a C++ service, robotics stack, or edge application;
- want to study or customize a model-owned builder and runtime together;
- want a small artifact boundary between the build environment and an
  application; or
- want task-oriented interfaces across text, vision, audio, diffusion,
  segmentation, forecasting, and other families.

## What it simplifies

| Traditional pain point | TRTMC boundary |
| --- | --- |
| Export and conversion gaps | A family builds TensorRT engines directly from its supported checkpoint. |
| Repeated application integration | Applications load a bundle and call an abstract Task API. |
| Model changes spread across the repository | Normal contributions stay inside one `families/<family>/` directory. |
| Python framework dependencies during native inference | The family DSO owns native preprocessing, engine execution, and postprocessing. |
| Unclear runtime ownership | `trtmc inspect` reports `family`, `task`, `backend`, and section names. |

Declared support is not proof that every model passed on every platform. Use
[Supported Models](../models-recipes/overview.md) for exact checkpoints and
evidence, then follow the owning family's tests.

## Trust boundary

TensorRT-Model-Connect is a reference implementation. Users are responsible
for trusting the checkpoints, bundles, native libraries, and local environment
they provide when building or running a model.
