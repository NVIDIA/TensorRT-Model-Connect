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

## Validation expectation

Quantization changes should be validated with a parity test that matches the task:

- Decoder models: logits, generated text, and stable top-k behavior.
- Diffusion models: image/video health checks and reference comparison where available.
- Audio models: waveform sanity and ASR round-trip or task-specific oracle.
- Time-series models: relative L2 and pointwise error thresholds.
