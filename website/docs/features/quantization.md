---
title: Quantization
---

Quantization is a family-owned build choice. The shared API carries the
requested string but does not normalize formats, calibrate models, map scales,
or promise that every family supports it.

```bash
python -m tensorrt_model_connect build MODEL \
  --precision fp16 \
  --quantization fp8 \
  --output model-fp8.bundle
```

The selected `families/<family>/model.py` must explicitly implement or reject
the exact combination of checkpoint, precision, topology, and quantization.
There is no shared quantization registry, ModelOpt adapter system, generic
calibration CLI, or legacy `--fp8`/scale-file fallback in the current build
core.

## Current ownership examples

- Qwen owns its FP8 calibration and graph insertion in
  `families/qwen/quantization.py`; its builder restricts FP8 to qualified Qwen3
  single-device and TP4 combinations.
- FLUX owns packaged FLUX.2 scales under `families/flux/data/` and rejects FP8
  for other FLUX variants.
- Wan2.2 TI2V owns its packaged profile and scale loading under
  `families/wan2_2_ti2v/`.
- Families without an implemented quantized path reject any value other than
  absent/`none`.

These examples describe source behavior, not blanket qualification. Use the
live family manifests and retained E2E evidence for an exact support claim.

## Mixed precision

`--fp32-layer INDEX` is repeatable and becomes `BuildRequest.fp32_layers`.
Layer indexes are intentionally family-local selectors. Each family documents,
validates, applies, or rejects them; shared code does not reinterpret the
indexes.

## Required evidence

A quantized support claim should retain:

1. exact source and checkpoint revisions;
2. base precision, quantization format, scales/calibration source, and topology;
3. exact build request and produced bundle;
4. family-specific graph or engine evidence that quantization was applied;
5. Task-appropriate parity and output-health results; and
6. matched hardware and raw measurements for a performance claim.

Parser acceptance, a successful bundle build, or a shared unit test alone is
not model qualification. Never weaken a family threshold to make a result pass.
