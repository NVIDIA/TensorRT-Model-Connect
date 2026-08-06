---
slug: /
title: TensorRT-Model-Connect
description: Build deployable TensorRT bundles from Python-first checkpoints and run them through a native C++ task API.
---

import ModelSupportInventory from '@site/src/components/ModelSupportInventory';
import Diagram from '@site/src/components/Diagram';

TensorRT-Model-Connect turns a Hugging Face or local checkpoint into a
deployable `.bundle` bundle, then runs that bundle through a native C++ task API.

If you are new to the project, your first goal is deliberately small:

> Prepare one supported environment, build Qwen3-0.6B, inspect its bundle, and
> run one deterministic NLP/text-generation inference.

The [Getting Started](getting-started/overview.md) section is the only required
entry path for that goal.

<Diagram
  src="/img/diagrams/trtmc-system-map.svg"
  alt="TensorRT-Model-Connect system map from a checkpoint through bundle construction and C++ runtime execution"
  caption="The bundle is the stable handoff between Python model construction and the native task runtime."
/>

## Choose your path

<div className="trtmc-card-grid">
  <div className="trtmc-card">
    <strong>Run your first inference</strong>
    Check prerequisites, prepare the environment, install the project, and
    complete one build-inspect-run loop.
  </div>
  <div className="trtmc-card">
    <strong>Learn the project</strong>
    Follow one ordered curriculum from inference fundamentals through
    multimodal models, optimization, and validation.
  </div>
  <div className="trtmc-card">
    <strong>Integrate an API</strong>
    Look up the CLI, Python builder/wrapper, C++ runtime, or experimental
    C-linkage subset.
  </div>
  <div className="trtmc-card">
    <strong>Understand the design</strong>
    Follow block and sequence diagrams from checkpoint resolution to build,
    bundle loading, GPU execution, and typed results.
  </div>
  <div className="trtmc-card">
    <strong>Contribute or extend</strong>
    Choose the owning family, runtime, config, validation, or external-kernel
    boundary before editing.
  </div>
  <div className="trtmc-card">
    <strong>Research a feature</strong>
    Read current feature contracts and the historical context that explains
    why they exist.
  </div>
</div>

| Goal | Start here |
| --- | --- |
| First successful NLP/text inference | [Getting Started](getting-started/overview.md) |
| Beginner-to-advanced curriculum | [Learn & Tutorials](learning-path.md) |
| Exact public interfaces | [API Reference](api/overview.md) |
| Software architecture and component ownership | [Architecture & Design](architecture/overview.md) |
| Contribution and extension recipes | [Contribute & Extend](extend/overview.md) |
| Feature behavior and design history | [Feature Reference & Context](features/overview.md) |

## The minimum mental model

| Term | Meaning |
| --- | --- |
| Checkpoint | Model config, weights, tokenizer/processor assets, and related metadata released by a training ecosystem. |
| TensorRT engine | A target-specific compiled execution plan. It is not the original checkpoint. |
| `.bundle` bundle | The artifact boundary between Python-first build logic and native runtime execution. |
| Family | The model-owned Python implementation that recognizes a checkpoint and builds its artifacts. |
| Pipeline | The native task implementation returned to an application after the bundle is loaded. |

Use the [Glossary](getting-started/glossary.md) whenever a term is unfamiliar;
you do not need to memorize it before starting.

## Current declared inventory

The following facts are generated from model-owned metadata in the current
checkout:

<ModelSupportInventory variant="facts" />

These are discovery counts, not proof that every declared model passed on your
hardware. Use [Model Support](getting-started/model-support.md) for evidence
levels and exact ownership.

## What comes after the first run

After Getting Started:

1. use [Inference Fundamentals](getting-started/inference-fundamentals.md) to
   explain what happened;
2. follow the [Learning Path](learning-path.md) from beginner to advanced;
3. use [Architecture & Design](architecture/overview.md) only when you need
   source-level responsibilities; and
4. treat [Feature Reference & Context](features/overview.md) as an archive and
   deep-reference section, not as the default reading path.
