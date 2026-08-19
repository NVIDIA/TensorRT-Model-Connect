---
title: Advanced Tutorial - Quantization and Runtime Knobs
---

import Diagram from '@site/src/components/Diagram';

This tutorial covers build-time precision, post-training quantization, runtime cache sizing, and backend selection.

## Learning objectives

By the end of this lab, you should be able to place a knob at build, load, or
request time; distinguish precision from quantization; and verify whether two
bundles use the same native or platform-specialized execution path before
comparing them.

Select the CLI before running an example:

```bash
export TRTMC=trtmc
# Source build inside the development container:
# export TRTMC=./build/trtmc
```

The FP8 example that reads `tests/e2e/...` requires a repository checkout and
must run from its root; the selected CLI may still come from an installed
wheel.

The advanced knobs change either:

- How the bundle is built.
- How the runtime loads the bundle.
- How a request uses runtime state.

<Diagram
  src="/img/diagrams/tutorials/advanced/knob-scopes.svg"
  alt="Three knob boundaries where build-time inputs produce a native or optimized bundle, load options create IPipeline, and request options affect a typed task call"
  caption="Build-time inputs shape the artifact inside the selected family model module; load options apply while creating IPipeline, and request options apply to the later task operation."
/>

The shared builder resolves one family and calls its `model.build()` once.
Most family modules run their native recipe directly. Qwen's module may match
the resolved model revision, active target, and effective public options
against its exact optimized profiles; no match runs Qwen's native recipe, while
a claimed profile owns the build and fails terminally.

## Precision

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-fp16.bundle \
  --precision fp16
```

Supported precision choices in the CLI are `fp32`, `fp16`, and `bf16`.

Precision changes the numeric type used by the engine. It affects memory, speed, and numerical behavior.
For this eligible dense Qwen3 checkpoint, explicitly requesting FP16 opts out
of its BF16 full-context native-KV contract; Qwen's `model.py` selects its
compatible family-local native graph instead.

| Precision | Typical use |
| --- | --- |
| `fp32` | Debugging or highest numerical conservatism. |
| `fp16` | Common GPU inference default. |
| `bf16` | Useful when model/backend support favors BF16 behavior. |

For Qwen requests that use its exact-profile selection, precision is also a
profile input. Changing it can cause a profile to start or stop matching, so
two builds that differ only in the CLI flag can still use different runtime
implementations. Other families interpret precision entirely inside their own
native recipe.

## Quantization

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-fp8.bundle \
  --quantize fp8 \
  --quant-calibration-samples 512
```

The current quantization surface accepts `fp8`, `int8`, `int8_sq`, `int4`, `int4_awq`, `nvfp4`, and `w4a8`. A family `model.py` can exclude weight patterns, provide calibration data, and select its family-specific calibration adapter directly.

Qwen2.5-VL and Qwen3-VL use image-plus-text calibration rather than the
generic causal-language-model adapter. See
[Qwen-VL calibration inputs](../../features/quantization.md#qwen-vl-calibration-inputs)
for the paired-manifest, image-directory, placeholder, and evidence
boundaries.

<Diagram
  src="/img/diagrams/tutorials/advanced/quantization-flow.svg"
  alt="Quantization flow choosing precomputed scales, a prequantized checkpoint, ModelOpt calibration, or dynamic scales before building a TensorRT engine and bundle"
  caption="Calibration is only one scale-source path; other formats reuse supplied or checkpoint-owned scales or select dynamic scale policy before the builder emits the engine and required metadata."
/>

Quantization is not just a compression flag. It is a contract between:

| Part | Responsibility |
| --- | --- |
| Family `model.py` | Exclude sensitive weights, supply calibration prompts or adapters, support family-specific scale collection. |
| Quantization registry | Interpret format names and format-specific policy. |
| Builder | Apply quantization and write required metadata/scales. |
| Runtime | Load and execute the resulting engine; it should not redo calibration. |

Qwen includes quantization format, calibration settings, and scale inputs in
its family-owned exact-profile decision. If the combination is not selected,
Qwen continues through its native recipe. Other family modules consume those
options directly without a shared provider probe.

## Reusing scales

```bash
$TRTMC build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2-fp8.bundle \
  --fp8-scales python/tensorrt_model_connect/models/flux/tests/data/flux2-fp8-scales.json
```

Use `--save-fp8-scales` when you want to reuse calibrated scales across builds.

## Dynamic KV cache

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-dynamic.bundle \
  --dynamic-kv-cache \
  --dynamic-kv-profile-rows 256,512,1024
```

At runtime, override the cache memory budget with:

```bash
$TRTMC run /tmp/qwen3-dynamic.bundle \
  --prompt "Summarize dynamic KV cache." \
  --kv-cache-size 512MB
```

Dynamic KV cache separates the bundle's compiled profiles from the session's
cache budget. During plugin construction, model-owned code converts that
budget into admitted decoder contexts and inference-state capacity. During a
request, the pipeline reads the state's preferred row count and chooses a
matching decoder context.

For eligible dense Qwen3 and Llama, `--dynamic-kv-cache` deliberately opts out
of the native full-context fixed-KV route and uses the compatible family-local
dynamic-KV recipe. A native full-context bundle rejects runtime `--kv-cache-size`; its
physical capacity is fixed to the model context and shared by prefill and
decode.

<Diagram
  src="/img/diagrams/tutorials/advanced/dynamic-kv-cache.svg"
  alt="Dynamic KV-cache flow where the model plugin applies a session budget to compiled decoder contexts and the pipeline later selects one using preferred cache rows from state"
  caption="The plugin admits fixed decoder contexts and allocates state within the session budget; at each step the pipeline, not ITrtModule, uses preferred_cache_rows() to choose a matching context."
/>

## Native backend DSO search

```bash
$TRTMC run /tmp/model.bundle \
  --prompt "Hello" \
  --backend-dir /opt/trtmc/backends
```

For a native bundle, the runtime also searches the executable directory and
loader paths. Use `--backend-dir` when testing a native backend DSO that is not
next to `trtmc`.

Native backend selection is part of deployment correctness. The runtime checks
TensorRT version and ABI metadata so a bundle built with one ABI is not
silently executed with an incompatible backend. An optimized bundle embeds
its implementation DSO and artifacts; it does not dispatch through the native
model/backend DSO chain, so `--backend-dir` does not select its runtime.

## Native TensorRT-RTX

Build an RTX-targeted bundle:

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-rtx.bundle \
  --rtx
```

Run with a runtime cache:

```bash
$TRTMC run /tmp/qwen3-rtx.bundle \
  --prompt "Hello" \
  --runtime-cache /tmp/trtmc-rtx.cache \
  --cuda-graphs
```

For a native TensorRT-RTX bundle, `--runtime-cache` stores JIT kernel cache data
for faster repeat runs and `--cuda-graphs` requests graph capture when the
backend supports it. Optimized implementations own their graph-capture policy;
do not assume that the native `--cuda-graphs` switch enables, disables, or
otherwise reproduces an optimized implementation's qualified path.

<Diagram
  src="/img/diagrams/tutorials/advanced/rtx-runtime.svg"
  alt="Native TensorRT-RTX runtime flow from an RTX-targeted bundle through BackendLoader and the RTX backend DSO to ITrtModule and the public pipeline"
  caption="Runtime cache and CUDA-graph settings affect the native RTX backend; they do not define an optimized implementation's private policy."
/>

## Advanced knob checklist

First run regular `trtmc inspect /tmp/model.bundle` and record whether the section
list contains `optimized_runtime.json`. Regular inspection proves the bundle
kind but does not decode the optimized implementation/profile fields. Changing
precision or quantization can switch between optimized and native builds, so
performance comparisons are valid only after confirming that both artifacts
use the same execution path.

When reporting a result, always include:

| Area | Values to report |
| --- | --- |
| Build | Model ID, precision, quantization format, max cache length, dynamic profiles, build GPU, TensorRT version. |
| Artifact | Bundle path, native or optimized kind, family, runtime strategy or optimized implementation/profile evidence, and section layout. |
| Load | Native backend DSO/search path or optimized implementation path, runtime cache path, CUDA graph policy, and config overrides. |
| Request | Prompt/input shape, max tokens or steps, sampling settings, image/video dimensions, audio sample rate, forecast horizon. |
| Hardware | GPU model, driver, CUDA, TensorRT runtime, container or host environment. |

## Self-check

1. Why does parser acceptance of `--quantize fp8` not prove model support?
2. When is `--runtime-cache` a file, and when can it be a materialization root?
3. What must you inspect before treating an A/B timing result as a backend or
   quantization comparison?

<details>
<summary>Check your answers</summary>

1. The selected family must apply the format to the intended graph regions and
   pass task parity/quality and performance gates; a generic parser cannot
   prove that.
2. Native TensorRT-RTX uses it as a JIT cache file. An optimized runtime can use
   it as the root for integrity-bound provider artifacts.
3. Confirm model/revision/config, bundle kind, native strategy or optimized
   provider/profile, section layout, runtime dependencies, input, timing
   boundary, and quality gate are comparable.

</details>

{/* Collaborative review anchor: batch 2. */}
