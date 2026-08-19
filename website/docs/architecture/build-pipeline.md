---
title: Build Pipeline
description: How a checkpoint is resolved, routed, compiled, and packaged.
---

import Diagram from '@site/src/components/Diagram';

The build pipeline accepts a Hugging Face model ID or local directory and writes
one `.bundle` bundle. The public entry points are:

- `trtmc build`, implemented by
  `python/tensorrt_model_connect/build_cli.py`;
- `tensorrt_model_connect.build()`, implemented by
  `python/tensorrt_model_connect/engine_builder.py`.

Both entry points resolve the model's owning family, import its `model.py`, and
call that module's `build()` exactly once.

## Authoritative build routing

<Diagram
  src="/img/diagrams/architecture/build-route-selection.svg"
  alt="Build-route decision tree resolving one family model module and calling its complete build recipe"
  caption="The shared path resolves and dispatches only. The selected family owns native engines, any exact optimized profile, and final bundle assembly."
/>

There is no central native/optimized router or legacy build fallback. A family
that supports multiple implementations owns that decision locally.

## 1. Resolve the source model

The builder resolves a repository ID or local directory and reads the model
metadata needed for family selection:

- Transformers-style checkpoints normally provide `config.json`;
- Diffusers checkpoints provide `model_index.json`;
- tokenizers, processors, weights, and family-specific assets remain source
  inputs rather than runtime dispatch identities.

`ModelConfig` in `python/tensorrt_model_connect/config.py` normalizes common
architecture fields, including supported nested text/language configuration
forms.

## 2. Resolve the owning family

Family discovery is descriptor-driven:

1. A full config uses `architecture_patterns`, aliases, and prefixes to bound
   candidates, then calls their required `matches(config)` function.
2. A string or `model_type` uses the same indexed metadata.
3. A Diffusers pipeline class uses descriptor
   `diffusion_pipeline_classes` only; it has no all-package fallback.

The resolver imports exactly `models.<id>.model`. Unknown inputs fail closed;
there is no `pkgutil` scan, package proxy, or manifest-selected Python module.

## 3. Select the build path

### Exact-qualified optimized implementation

Qwen's `model.py` can probe its exact optimized implementation profiles before
executing its native recipe. Selection binds:

- model ID and immutable revision;
- active deployment target;
- effective public build options;
- implementation and profile identity;
- profile `qualification_state` and semantic-source digest.

Exactly one profile may claim the request. The isolated adapter produces
opaque runtime artifacts and an exact `libtrtmc_impl_*.so`; the shared packager
writes those artifacts with `optimized_runtime.json`.

Those profile fields drive exact selection; they are not a fresh
target-hardware result. The public Source tree does not publish the former
qualification runner or retained target artifacts.

### Native family recipe

Each `model.py` loads weights, applies its config and quantization policy,
constructs one or more TensorRT networks, compiles engine plans, assembles
model-specific sections, and writes the bundle.

Supplying a trusted external-kernel manifest with `--kernel` deliberately uses
the owning family's native TensorRT path rather than optimized-provider
selection.

## Native build units

| Unit | Responsibility |
| --- | --- |
| `ModelConfig` | Normalize source configuration |
| Family `model.py` | Match the model and own config → weights → engines → bundle |
| Family checkpoint mapper | Translate source tensor names and layouts |
| Family graph helpers/builders | Express model-specific TensorRT graph semantics |
| Quantization units | Plan calibration, scales, formats, and exclusions |
| `BundleInfo` / `BundleSection` | Describe native metadata and ordered payloads |
| `write_bundle()` | Atomically serialize the final `.bundle` |

Graph helpers stay under their owning family. There is no supported
repository-root `graph_ops.py`, `graph_blocks.py`, or one-size-fits-all decoder
builder contract.

## Shape and optimization-profile contract

TensorRT optimization profiles are part of the built artifact. They constrain
which input shapes and execution modes are legal at run time.

- A decoder may use separate prefill/decode engines or one dual-profile engine.
- Dynamic-KV and fixed full-context bundles have different cache constraints.
- Diffusion, audio, vision, and time-series families use different component
  engines and batch policies.
- An optimized implementation owns its own profile and shape contract behind
  the public pipeline API.

Do not infer a universal engine layout from the model modality or from another
family. Inspect the bundle, its owning descriptor, and the exact build options.

## Build-time configuration

Build `--config` and `--set` inputs are validated through Python schema
definitions and can feed supported builder arguments. They should not be
confused with run-time session overrides.

The general config model defines several provenance layers, but an ordinary
native bundle does not automatically persist every build CLI contribution as a
run-time bundle default. A producer must explicitly write a top-level
`defaults` object for that behavior.

## Output boundary

Both routes call the common bundle serializer, but their payloads differ:

- native: `runtime_strategy`, `config.json`, TensorRT plan sections, tokenizer
  or processor assets, and family metadata;
- optimized: `optimized_runtime.json`, private implementation metadata, and an
  integrity-bound artifact tree containing the exact implementation DSO.

Continue with [Bundle Format](bundle-format.md) for the physical container and
[Runtime Lifecycle](runtime-lifecycle.md) for load-time behavior.

{/* Collaborative review anchor: batch 2. */}
