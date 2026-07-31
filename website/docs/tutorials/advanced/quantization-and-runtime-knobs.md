---
title: Advanced Tutorial - Quantization and Runtime Knobs
---

import Diagram from '@site/src/components/Diagram';

This tutorial covers build-time precision, post-training quantization, runtime cache sizing, and backend selection.

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
  caption="Build-time inputs shape the artifact, although a family native-default gate sees only ModelConfig; load options apply while creating IPipeline, and request options apply to the later task operation."
/>

Family routing runs before the optimized-runtime probe. Eligible dense Qwen3
and Llama checkpoints declare a native default and skip that probe. For other
requests, the probe matches the resolved model revision, active target, and
effective public build options. A single supported profile owns the build; no
supported profile falls back to the native path. A build failure after a
profile claims the request is terminal.

## Precision

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-fp16.trtfb \
  --precision fp16
```

Supported precision choices in the CLI are `fp32`, `fp16`, and `bf16`.

Precision changes the numeric type used by the engine. It affects memory, speed, and numerical behavior.
For this eligible dense Qwen3 checkpoint, explicitly requesting FP16 keeps the
family's native route but opts out of its BF16 full-context native-KV contract;
the family uses its legacy native builder instead.

| Precision | Typical use |
| --- | --- |
| `fp32` | Debugging or highest numerical conservatism. |
| `fp16` | Common GPU inference default. |
| `bf16` | Useful when model/backend support favors BF16 behavior. |

For families that reach provider probing, precision is also an
optimized-profile input. Changing it can cause a profile to start or stop
matching, so two builds that differ only in the CLI flag can still use different
runtime implementations. Dense Qwen3 and Llama architectures that claim their
native default do not re-enter provider selection when precision changes.

## Quantization

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-fp8.trtfb \
  --quantize fp8 \
  --quant-calibration-samples 512
```

The current quantization surface accepts `fp8`, `int8`, `int8_sq`, `int4`, `int4_awq`, `nvfp4`, and `w4a8`. Family plugins can exclude weight patterns, provide calibration data, and return a family-specific calibration adapter through the `FamilyPlugin` protocol.

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
| Family plugin | Exclude sensitive weights, supply calibration prompts or adapters, support family-specific scale collection. |
| Quantization registry | Interpret format names and format-specific policy. |
| Builder | Apply quantization and write required metadata/scales. |
| Runtime | Load and execute the resulting engine; it should not redo calibration. |

For requests that reach provider probing, quantization format, calibration
settings, and scale inputs are forwarded as public build options. If the exact
combination is not qualified by an optimized implementation, the command
proceeds through the native builder instead. The dense Qwen3 example above
already owns the native route, so quantization opts into its compatible legacy
native builder without probing a provider.

## Reusing scales

```bash
$TRTMC build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2-fp8.trtfb \
  --fp8-scales tests/e2e/models/flux/data/flux2-fp8-scales.json
```

Use `--save-fp8-scales` when you want to reuse calibrated scales across builds.

## Dynamic KV cache

```bash
$TRTMC build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-dynamic.trtfb \
  --dynamic-kv-cache \
  --dynamic-kv-profile-rows 256,512,1024
```

At runtime, override the cache memory budget with:

```bash
$TRTMC run /tmp/qwen3-dynamic.trtfb \
  --prompt "Summarize dynamic KV cache." \
  --kv-cache-size 512MB
```

Dynamic KV cache separates the bundle's compiled profiles from the session's
cache budget. During plugin construction, model-owned code converts that
budget into admitted decoder contexts and inference-state capacity. During a
request, the pipeline reads the state's preferred row count and chooses a
matching decoder context.

For eligible dense Qwen3 and Llama, `--dynamic-kv-cache` deliberately opts out
of the native full-context fixed-KV route and uses the compatible legacy
builder. A native full-context bundle rejects runtime `--kv-cache-size`; its
physical capacity is fixed to the model context and shared by prefill and
decode.

<Diagram
  src="/img/diagrams/tutorials/advanced/dynamic-kv-cache.svg"
  alt="Dynamic KV-cache flow where the model plugin applies a session budget to compiled decoder contexts and the pipeline later selects one using preferred cache rows from state"
  caption="The plugin admits fixed decoder contexts and allocates state within the session budget; at each step the pipeline, not ITrtModule, uses preferred_cache_rows() to choose a matching context."
/>

## Native backend DSO search

```bash
$TRTMC run /tmp/model.trtfb \
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
  -o /tmp/qwen3-rtx.trtfb \
  --rtx
```

Run with a runtime cache:

```bash
$TRTMC run /tmp/qwen3-rtx.trtfb \
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

First run regular `trtmc inspect /tmp/model.trtfb` and record whether the section
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
