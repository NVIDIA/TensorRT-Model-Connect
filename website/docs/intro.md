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
    <strong>Get started</strong>
    Check prerequisites, prepare the environment, install the project, and
    complete one build-inspect-run loop.
  </div>
  <div className="trtmc-card">
    <strong>Find a model</strong>
    Search exact checkpoint IDs, tasks, family ownership, precision, and
    topology configurations generated from manifests.
  </div>
  <div className="trtmc-card">
    <strong>Use a feature</strong>
    Look up task commands, build behavior, runtime configuration,
    quantization, parallelism, validation, and benchmarking.
  </div>
  <div className="trtmc-card">
    <strong>Follow a course</strong>
    Learn progressively through short modules, hands-on labs, milestones,
    self-check questions, and answer keys.
  </div>
  <div className="trtmc-card">
    <strong>Look up a contract</strong>
    Open the CLI, Python, C++, bundle, configuration, testing, or performance
    reference without reading a tutorial first.
  </div>
  <div className="trtmc-card">
    <strong>Develop or contribute</strong>
    Understand architecture and choose the owning model, runtime, config,
    validation, or optimized-provider boundary before editing.
  </div>
</div>

| Goal | Start here |
| --- | --- |
| Understand the project boundary and intended users | [Project Overview](getting-started/project-overview.md) |
| First successful NLP/text inference | [Getting Started](getting-started/overview.md) |
| Exact model/checkpoint lookup | [Models & Recipes](models-recipes/overview.md) |
| Task and feature lookup | [User Guides](user-guides/overview.md) |
| Beginner-to-advanced curriculum | [Tutorials](learning-path.md) |
| Exact public interfaces and configuration | [Reference](api/overview.md) |
| Architecture, extension, and contribution | [Developer Guide](developer-guide/overview.md) |
| Compatibility and lifecycle policy | [Release & Support](release-support/overview.md) |
| Machine-readable and agent safety guidance | [AI & Agent Guide](agent-guide.md) |

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
hardware. Use [Model Support](models-recipes/overview.md) for evidence
levels and exact ownership.

## What comes after the first run

After Getting Started:

1. use [Inference Fundamentals](getting-started/inference-fundamentals.md) to
   explain what happened;
2. use [User Guides](user-guides/overview.md) for the next task you need to do;
3. follow the [Tutorial Curriculum](learning-path.md) when you want progressive
   learning and self-checks; and
4. use [Developer Guide](developer-guide/overview.md) only when you need
   source-level ownership or contribution instructions.

{/* Collaborative review anchor. */}
