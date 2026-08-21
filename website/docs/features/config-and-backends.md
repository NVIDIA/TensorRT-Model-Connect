---
title: Config and Backends
---

The CLI examples use one selector for an installed wheel or source build:

```bash
export TRTMC=trtmc
# Source build inside the development container:
# export TRTMC=./build/trtmc
```

## Schema-driven config

Both build and runtime CLIs expose a generic config surface:

```bash
--config profile.json
--set namespace.field=value
```

The goal is to add native build/runtime feature knobs through registered
schemas instead of growing custom CLI flags for every feature. Optimized
implementations receive the public option tuple through their family-owned
adapter contract; the generic router does not reinterpret those options.

Schema sources live under:

- `src/runtime/config/schemas/`
- `include/trtmc/config/schemas/`
- `python/tensorrt_model_connect/runtime_config/schemas/`
- `cmake/trtmc_config_schemas.cmake`

Model-owned schemas instead live beside their owners:

- Python: `python/tensorrt_model_connect/families/<family>/runtime_config_schema.py`
- C++: `src/runtime/models/<owner>/config_schema.cpp`
- Registration: the owner's `runtime_config_schemas` entry in
  `src/runtime/models/<owner>/MODEL.toml`

## Live schema catalog

This is the current registered namespace/field inventory.

| Namespace and owner | Registered fields | Allowed layers |
| --- | --- | --- |
| `platform` (shared host/runtime) | `platform.source_dir`, `platform.trt_log_stderr`, `platform.trt_log_min_severity` | Session request, platform profile |
| `runtime` (shared decode runtime) | `runtime.disable_cuda_graph`, `runtime.prefer_gpu_greedy` | Session request, platform profile |
| `text_trace` (shared text diagnostics) | `text_trace.step_trace_path`, `text_trace.step_trace_start_pos`, `text_trace.step_trace_end_pos`, `text_trace.step_trace_topk` | Session request, platform profile |
| `triattention` (shared KV compaction) | `triattention.enabled`, `triattention.kv_budget`, `triattention.divide_length`, `triattention.recent_window`, `triattention.score_aggregation`, `triattention.per_layer_aggregation`, `triattention.count_prompt_tokens`, `triattention.protect_prefill`, `triattention.disable_mlr`, `triattention.disable_trig`, `triattention.offset_max_length`, `triattention.stats_section`, `triattention.debug`, `triattention.profile`, `triattention.runtime_bucket_rows`, `triattention.disable_gpu_selection`, `triattention.disable_gpu_compaction`, `triattention.disable_gpu_state`, `triattention.zero_tail`, `triattention.dump_keep_path`, `triattention.dump_compaction_index`, `triattention.abort_after_dump`, `triattention.dump_score_cache`, `triattention.dump_score_values` | Core fields: bundle default/session; `stats_section`: build/bundle; diagnostics: session |
| `audio_bark` (Bark) | `audio_bark.dump_path`, `audio_bark.greedy`, `audio_bark.seed`, `audio_bark.fine_temperature` | Session request, platform profile |
| `audio_magpie` (Magpie TTS) | `audio_magpie.greedy`, `audio_magpie.cfg_scale`, `audio_magpie.temperature`, `audio_magpie.finished_limit`, `audio_magpie.seed`, `audio_magpie.max_source_positions` | Sampling fields: session/platform; `max_source_positions`: build/bundle |
| `wan2_2_ti2v` (Wan 2.2 TI2V) | `wan2_2_ti2v.easycache_enabled`, `wan2_2_ti2v.easycache_threshold`, `wan2_2_ti2v.easycache_first_exact_steps`, `wan2_2_ti2v.easycache_last_exact_steps`, `wan2_2_ti2v.easycache_max_consecutive_reuse`, `wan2_2_ti2v.late_cfg_enabled` | Session request, platform profile |
| `sana_wm` (SANA-WM) | `sana_wm.image_path`, `sana_wm.action`, `sana_wm.translation_speed`, `sana_wm.rotation_speed_deg`, `sana_wm.num_frames`, `sana_wm.fps`, `sana_wm.flow_shift`, `sana_wm.intrinsics`, `sana_wm.no_refiner` | Session request, platform profile |
| `qwen_vl_decoder` (Qwen-VL build) | `qwen_vl_decoder.decode_attention`, `qwen_vl_decoder.max_prefill_length`, `qwen_vl_decoder.opt_prefill_length`, `qwen_vl_decoder.builder_workspace_gib` | Build time, bundle default, build CLI request |
| `qwen_vl_lora` (Qwen-VL build) | `qwen_vl_lora.enabled`, `qwen_vl_lora.max_rank`, `qwen_vl_lora.target_modules` | Build time, bundle default, build CLI request |
| `qwen_vl_vision` (Qwen-VL build) | `qwen_vl_vision.image_height`, `qwen_vl_vision.image_width`, `qwen_vl_vision.dynamic_resolution`, `qwen_vl_vision.min_pixels`, `qwen_vl_vision.opt_pixels`, `qwen_vl_vision.max_pixels` | Build time, bundle default, build CLI request |

The Qwen-VL build schemas have these exact defaults and validation rules:

| Field | Type and default | Additional contract |
| --- | --- | --- |
| `qwen_vl_decoder.decode_attention` | `string`, `native` | Accepts `native` or `decomposed`; decomposed decode is valid only for an active split-decoder build. |
| `qwen_vl_decoder.max_prefill_length` | `int32`, `0` | Nonnegative; zero lets the builder use the cache-length bound for the prefill maximum. |
| `qwen_vl_decoder.opt_prefill_length` | `int32`, `64` | Must be positive and is clamped to the effective prefill maximum. |
| `qwen_vl_decoder.builder_workspace_gib` | `int32`, `0` | Nonnegative; zero preserves TensorRT's device default, while a positive value caps decoder builder workspace independently of the cache bound. |
| `qwen_vl_lora.enabled` | `bool`, `false` | Dynamic binding currently supports Qwen2.5-VL only and rejects tensor-parallel builds. |
| `qwen_vl_lora.max_rank` | `int32`, `0` | Schema range is 0 through 256; enabling LoRA requires 1 through 256. |
| `qwen_vl_lora.target_modules` | `string`, `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Comma-separated non-empty subset of exactly those seven projection names. |
| `qwen_vl_vision.image_height` | `int32`, `448` | Positive and divisible by the model's `patch_size * spatial_merge_size`. Rectangular profiles currently support Qwen2.5-VL only. |
| `qwen_vl_vision.image_width` | `int32`, `448` | Same alignment and family restrictions as height. |
| `qwen_vl_vision.dynamic_resolution` | `bool`, `false` | Enables Qwen smart-resize and a dynamic vision profile for Qwen2.5-VL. Qwen3-VL currently rejects this mode. |
| `qwen_vl_vision.min_pixels` | `int32`, `0` | Nonnegative. Zero selects `preprocessor_config.json` when present, otherwise the builder falls back to 3136 pixels. |
| `qwen_vl_vision.opt_pixels` | `int32`, `200704` | Positive and must satisfy effective `min_pixels <= opt_pixels <= max_pixels`. This is the dynamic TensorRT profile's optimum. |
| `qwen_vl_vision.max_pixels` | `int32`, `0` | Nonnegative. Zero selects `preprocessor_config.json` when present, otherwise the builder falls back to 12845056 pixels. |

The dynamic profile preserves the source aspect ratio while aligning dimensions
to `patch_size * spatial_merge_size`; inputs with an aspect ratio greater than
200 are rejected. The packaged `preprocessor_config.json` remains the runtime
authority for `min_pixels` and `max_pixels` when it declares those fields, so
do not interpret explicit build values as runtime overrides of packaged
preprocessor metadata.

These are build settings even though the build CLI currently contributes
`--config` and `--set` values at `SessionRequest` priority before forwarding
them to the family builder. They do not imply that a loaded optimized runtime
accepts runtime config overrides.

Build routing happens before the native schema-driven builder: `trtmc build`
first resolves the family and asks whether its model-owned
`default_build_route` accepts the checkpoint. Eligible dense Qwen3 and Llama
take that native route directly. Otherwise the model, revision, target, and
public option tuple is offered to the family's exact qualified optimized
profiles. One claim owns the option semantics; no claim continues to the native
builder, which resolves registered schemas and rejects unknown namespaces,
fields, or invalid values.

At runtime, the two paths differ:

| Bundle path | Config behavior |
| --- | --- |
| Native | `ConfigBundle` can represent `SessionRequest > PlatformProfile > BundleDefault > BuildTime > SchemaDefault`. The current `PipelineFactory` passes the materialized `config.json` section to the resolver. A top-level `defaults` object there can contribute `BundleDefault`; load options can add `SessionRequest`. |
| Optimized | `optimized_runtime.json` claims the bundle before native config/plugin/backend dispatch. The embedded implementation receives `LoadOptions` through its private factory request and decides which options it supports; the current Qwen Edge-LLM implementation rejects runtime `--config`/`--set`. |

The five-layer order is the general resolver model, not a claim that every
factory call currently injects five independent contributions. In the current
native factory:

- schema defaults are supplied by the registered schema;
- a top-level `defaults` object in the materialized `config.json` section can
  contribute `BundleDefault`;
- the native builders do not currently add that object automatically, so
  normal builder-produced bundles usually contribute no `BundleDefault`;
- `BundleInfo.defaults` in the binary header is not passed to runtime config
  resolution;
- `LoadOptions.config_path` and CLI `--config`/`--set` all contribute
  `SessionRequest`;
- no separate `BuildTime` contribution is injected;
- no separate `PlatformProfile` contribution is injected.

Callers that build a `ConfigBundle` directly may use the other allowed layers.
Do not infer `BuildTime` or `PlatformProfile` provenance merely because a
config file contains build- or machine-specific values.

The C++ CLI pre-validates explicit `--config`/`--set` values against registered
schemas and exits nonzero on invalid input. That validation does not turn the
result into a native `ConfigBundle` for an optimized implementation.

Direct `PipelineFactory` callers on the native path have a different current
error behavior: the factory catches a resolution error, prints
`[trtmc.config] Failed to resolve runtime config`, and continues with
`runtime_config == nullptr`; the owning native plugin then chooses its local
fallback behavior. Successful resolution writes
`<bundle>.effective_config.json`; a failed factory resolution does not write a
new effective-config file. Callers using the factory API must treat that
warning as an error if silent fallback is unacceptable.

## Native backend DSOs

The native runtime path loads TensorRT backends dynamically:

- Standard TensorRT backend: `libtrtmc_backend_trt.so`
- ABI-suffixed standard backend alias when available: `libtrtmc_backend_trt_<major>_<minor>.so`
- TensorRT-RTX backend: `libtrtmc_backend_trt_rtx.so`

Use `--backend-dir` to add explicit backend search directories:

```bash
$TRTMC run /tmp/model.bundle \
  --prompt "Hello" \
  --backend-dir /opt/trtmc/backends
```

An optimized bundle bypasses this selection. The host materializes the
integrity-bound artifact tree, loads its exact embedded
`libtrtmc_impl_*.so`, and lets that implementation own downstream runtime
dependencies. `--backend-dir` is not a generic optimized-runtime DSO search
path.

## Runtime cache and CUDA graphs

The same public options have path-specific ownership:

| Option | Native path | Optimized path |
| --- | --- | --- |
| `--runtime-cache` | TRT-RTX JIT cache path passed to the native plugin/backend. | Root directory where the generic host materializes the integrity-bound optimized artifact tree. |
| `--cuda-graphs` | Passed to the native plugin/backend; use only when supported. | Forwarded in `LoadOptions`; the embedded implementation decides whether it supports or rejects the option. |

For a native TRT-RTX bundle:

```bash
--runtime-cache /tmp/trtmc-rtx.cache
--cuda-graphs
```

For an optimized bundle, use a directory as the cache root rather than an RTX
cache-file name:

```bash
$TRTMC run /tmp/optimized.bundle \
  --prompt "Hello" \
  --runtime-cache /tmp/trtmc-optimized-cache
```

Check the selected implementation's qualification contract before passing
provider-specific options such as CUDA graphs.

{/* Collaborative review anchor: batch 2. */}
