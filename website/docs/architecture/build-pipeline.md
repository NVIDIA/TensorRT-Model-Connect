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

Both entry points resolve the model's owning family before committing to a
native or optimized implementation.

## Authoritative build routing

<Diagram
  src="/img/diagrams/architecture/build-route-selection.svg"
  alt="Build-route decision tree resolving a model family, accepting its native default or probing exact optimized profiles, then writing one bundle"
  caption="A family-owned native default evaluates only the resolved ModelConfig and bypasses optimized-profile probing. Otherwise target and effective public options enter the exact-profile probe; no match uses native and ambiguity fails closed."
/>

Multiple optimized profiles claiming one request are an error. Once an
optimized adapter has claimed a request, its build failure is terminal; the
router does not silently retry the native path.

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

Family discovery is descriptor-driven, but it has three intentionally distinct
flows:

1. A full config uses `architecture_patterns` to bound candidates, evaluates
   their `matches_config()` predicates, then uses the all-package compatibility
   fallback only when needed.
2. A string or `model_type` tries a direct descriptor ID, alias/prefix
   candidates, then the same compatibility fallback.
3. A Diffusers pipeline class uses descriptor
   `diffusion_pipeline_classes` only; it has no all-package fallback.

Descriptor discovery imports the selected family package and reads its
package-level `plugin`. The descriptor's `module` field is specialization and
tooling metadata, not an arbitrary import selector.

## 3. Select the build path

### Model-owned native default

A family can declare a `default_build_route`. When that route accepts the
resolved `ModelConfig`, it owns the native build immediately. The callable does
not receive public build options; those options still affect the selected
native builder, but enter optimized profile matching only when the native
default does not claim the model. This is model-owned policy, not a central
model-name switch.

### Exact-qualified optimized implementation

Requests without a matching native default probe implementations only inside
the selected family. Selection binds:

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

### Native fallback

If no optimized profile claims the request, the native `FamilyPlugin` owns the
rest of the build. It loads weights, applies family config and quantization
policy, constructs one or more TensorRT networks, compiles engine plans, and
emits native bundle metadata.

Supplying a trusted external-kernel manifest with `--kernel` deliberately uses
the owning family's native TensorRT path rather than optimized-provider
selection.

## Native build units

| Unit | Responsibility |
| --- | --- |
| `ModelConfig` | Normalize source configuration |
| `FamilyPlugin` | Match the model, load weights, and own build entry points |
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
