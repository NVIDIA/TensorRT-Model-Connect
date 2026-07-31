---
title: Bundle Format
description: The physical and semantic contract carried by a .trtfb artifact.
---

import Diagram from '@site/src/components/Diagram';

`.trtfb` is the build/run boundary of TensorRT-Model-Connect. It is a
self-describing container with a JSON header and named binary sections.

Two payload shapes use the same outer format:

| Shape | Bundle owns | Runtime installation owns |
| --- | --- | --- |
| Native | `runtime_strategy`, `config.json`, TensorRT plans, tokenizer/processor assets, family metadata | Core runtime, owning model DSO, compatible backend DSO |
| Optimized | `optimized_runtime.json`, private implementation metadata, integrity-bound artifact tree, exact implementation DSO | Core host plus compatible driver/CUDA/TensorRT/loader/system libraries |

## Physical layout

<Diagram
  src="/img/diagrams/architecture/bundle-layout.svg"
  alt="Physical model.trtfb layout with magic bytes, JSON header length, metadata and section table, then native or optimized binary payloads"
  caption="The fixed prefix locates the JSON header; its section table addresses named entries inside the contiguous payload area."
/>

The exact magic bytes and length encoding are shared by:

- `python/tensorrt_model_connect/bundle_writer.py`;
- `src/bundle/bundle_format.h`;
- `src/bundle/bundle_format.cpp`.

The section offsets in the JSON table are relative to the beginning of the
binary payload area, after the header.

## Header metadata

The public C++ `BundleInfo` in `include/trtmc/bundle.h` exposes:

| Field group | Meaning |
| --- | --- |
| `model_id`, `model_type`, `family` | Source and builder identity |
| `precision` | Recorded build precision |
| `trt_version`, `trt_abi`, `gpu_name` | Build/runtime compatibility metadata |
| `created_at` | Producer timestamp when recorded |
| `vocab_size`, `hidden_size`, `num_layers` | Common model dimensions |
| `num_attention_heads`, `num_key_value_heads` | Common attention dimensions |
| `max_cache_length` | Recorded default cache capacity for decoder-like artifacts |
| `runtime_strategy` | Native dispatch key; empty is valid for optimized bundles |
| `tokenizer_add_special_tokens`, `tokenizer_add_special_tokens_present` | Tokenizer compatibility metadata |
| `sections` | Public name/offset/size inventory |
| `max_batch_size` | Per-component diffusion batch envelope |

Not every field applies to every modality. Consumers should use section and
capability checks rather than infer a decoder layout from zero/default-valued
metadata.

## Native payload

A native bundle normally contains `config.json` plus model-owned plans and
assets. Common section shapes include:

| Task shape | Typical sections |
| --- | --- |
| Decoder text generation | Prefill/decode or dual-profile engine plans, tokenizer assets, config |
| Vision-language | Decoder `engine_plan`, optional split `prefill_engine_plan`, optional vision plan, image-processing metadata, tokenizer assets |
| Diffusion/image/video | Text encoder, denoiser, VAE, scheduler, and latent metadata |
| Speech | Encoder/decoder or RNNT plans, filterbank/mel metadata, tokenizer assets |
| Audio generation | Semantic, acoustic, codec, tokenizer/phoneme, and audio metadata |

These names are examples, not a universal required set. The owning
`IPipelinePlugin` validates the exact sections its strategy consumes.
Graph-patched native bundles additionally carry `kernel_slots.json` for their
load-time TVM-FFI binding contract; the external kernel DSO is not embedded.
For example, current Qwen2.5-VL can emit split prefill/decode sections for a
supported single-GPU fixed-cache request, while Qwen3-VL, tensor-parallel,
dynamic-KV/TriAttention, or explicit dual-profile requests use another actual
layout. The bundle's `config.json.decoder_engine_layout` and section table
record the result.

For native dispatch, `runtime_strategy` is the critical identity. It selects
one model owner in the generated runtime index; loading that model DSO
registers the concrete `IPipelinePlugin`.

`task_strategy` is not the bundle dispatch key. It belongs to E2E task
orchestration and may group several model-owned native strategies.

## Optimized payload

An optimized bundle contains:

- `optimized_runtime.json`, with implementation, profile, model, target,
  factory, metadata-section, and artifact-tree identities;
- private implementation metadata, commonly in `implementation.json`; and
- `optimized_runtime_artifacts/...`, containing the exact implementation DSO
  and provider-produced runtime artifacts.

`config.json` is optional for this shape. The generic host does not use it to
select a native strategy.

`PipelineFactory` checks the header for `optimized_runtime.json` before native
bundle materialization. Descriptor presence claims the optimized path. The host
validates bounded identities and paths, verifies the artifact-tree digest,
materializes the tree in the runtime cache, loads only the embedded
implementation DSO, validates its private factory, and asks it for an
`IPipeline`.

There is no native fallback after that descriptor is present.

## Header versus `config.json`

The header is intended for fast inspection and section lookup without creating
a pipeline. Native `config.json` carries richer strategy-specific construction
data such as:

- `runtime_strategy`;
- engine/backend selection metadata;
- IO maps and tensor names;
- family-specific fields;
- an optional top-level `defaults` object for schema-controlled bundle
  defaults.

When metadata exists in both places, code should use the same precedence and
compatibility rules as `PipelineFactory`, not invent a separate reader policy.

## Sidecar files are not sections

Successful schema-driven configuration resolution may write
`<bundle>.effective_config.json` beside a bundle. Build timing and validation
reports may also be stored nearby.

Those files are diagnostics/evidence, not `.trtfb` sections. Moving the bundle
does not move its sidecars automatically.

## Inspection

Use the public C++ API to inspect metadata without constructing a pipeline:

```cpp
#include <trtmc/bundle.h>

auto info = trtmc::InspectBundle("/tmp/model.trtfb");
bool has_magic = trtmc::IsBundle("/tmp/model.trtfb");
```

The CLI provides the same user workflow:

```bash
trtmc inspect /tmp/model.trtfb
trtmc inspect /tmp/model.trtfb --list-engines
```

`--list-engines` recognizes native plan naming conventions. Optimized artifacts
use implementation-owned paths, so an optimized bundle can be valid even when
that command reports no native engine sections.

Inspection proves that the container and metadata can be read. It does not load
the model/backend or implementation DSO and does not prove inference.

## Compatibility and security boundaries

- Native engine plans require a compatible TensorRT runtime/backend ABI.
- Model DSOs and backend DSOs must be discoverable from the installation or
  explicit search paths.
- Optimized artifacts are content-addressed and bounded before
  materialization; unsafe paths, unexpected identities, or digest mismatches
  fail closed.
- The host never substitutes an installed same-name optimized implementation
  DSO.
- Neither bundle shape includes the complete driver, CUDA, TensorRT, dynamic
  loader, or operating-system environment.
- Legal runtime shapes remain constrained by the optimization profiles built
  into the selected plans or implementation.

## Load-time TVM-FFI slots

A graph-patched native bundle stores the fixed TensorRT engine plus a strict
`kernel_slots.json` section. V1 contains exactly one slot and records its ID
and ABI SHA-256; the runtime derives the plugin name as `trtmc.slot.<id>`.
The external kernel DSO is not stored in the bundle.

At pipeline load time, the CLI `--kernel-bindings` option or the additive C++
`trtmc::load(bundle, options, bindings_path)` overload supplies a strict JSON
manifest. The bundle slot must appear exactly once with the same ID and ABI
SHA-256, a relative DSO path, and the exported TVM-FFI module function. Unknown
or missing fields, extra or missing bindings, ABI mismatches, and unresolved
functions fail the load.

The engine ABI does not change when a DSO is selected. The same bundle can
create a new pipeline with a different ABI-compatible DSO. A pipeline keeps
the function acquired during its own load; there is no in-place rebind of a
running pipeline. Load-time slots apply only to native TensorRT bundles and
are rejected for optimized-runtime bundles.
V1 also fails closed when a model defers engine deserialization beyond the
pipeline-load call, and the current C-linkage entrypoints do not expose a
kernel-binding parameter.

Continue with [Runtime Lifecycle](runtime-lifecycle.md) to see how these
identities drive pipeline construction.
