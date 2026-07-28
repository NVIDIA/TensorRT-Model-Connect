---
title: Quantization
---

Quantization is a build-time feature controlled by the Python builder and family plugins.

## CLI surface

```bash
./build/trtmc build <model> -o <bundle.trtfb> --quantize fp8
```

Supported `--quantize` values are:

```text
fp8, int8, int8_sq, int4, int4_awq, nvfp4, w4a8
```

The builder also exposes FP8-specific flags:

```bash
--fp8
--fp8-scales <path>
--save-fp8-scales <path>
```

## Family controls

Family plugins can override:

- `quant_exclude_patterns()`
- `calibration_data()`
- `quant_adapter()`
- `fp8_precomputed_scales()`
- `fp8_calibrate()`

For `--fp8`, a family-provided precomputed scale profile is preferred before
live calibration. These hooks keep quantization policy and qualification close
to the model architecture that needs it.

## Build-path selection

`trtmc build` resolves the owning family first. If the family declares a
matching model-owned native default, the request goes directly to the native
TensorRT builder; eligible dense Qwen3 and Llama currently do this. Other
requests probe that family's optimized-runtime implementations for an exact
match on the model revision, active target, and effective public build options.
A single supported profile produces an optimized bundle. If no profile claims
the request, the command falls back to the native builder. Once a profile
claims the request, a failure in its build is terminal rather than a silent
native fallback.

Precision and quantization are part of those effective build options. Changing
`--precision`, `--quantize`, calibration settings, or scales can therefore do
more than change engine numerics: it can make an optimized profile match or
stop matching and switch the resulting bundle to the native path.

After each build, use regular `trtmc inspect <bundle.trtfb>` output to verify
the section layout. An `optimized_runtime.json` section identifies the
optimized path; regular inspection lists that section but does not decode its
implementation or profile fields. Record the bundle kind with parity and
performance results. Compare performance only when both bundles use the same
execution path; otherwise the result mixes a precision or quantization change
with a runtime implementation change.

## Validation expectation

Quantization changes should be validated with a parity test that matches the task:

- Decoder models: logits, generated text, and stable top-k behavior.
- Diffusion models: image/video health checks and reference comparison where available.
- Audio models: waveform sanity and ASR round-trip or task-specific oracle.
- Time-series models: relative L2 and pointwise error thresholds.
