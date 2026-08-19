---
title: Quantization
---

Quantization is a build-time optimization. The shared framework normalizes a
request, scales, and graph operations, while the selected model family owns
the decision to support that format and the evidence that it works.

:::warning A CLI value is not a support claim

The parser accepts every registered format for every model. A family build can
also accept a `quant_ctx` argument without applying it to every graph path.
Neither parser acceptance, bundle metadata, nor a successful build proves that
quantization was applied correctly. Verify the selected family's graph,
bundle, task output, and performance before making a support claim.

:::

## Generic quantization path

Use `--quantize` for families integrated with the shared quantization
framework:

```bash
trtmc build /path/to/model \
  --quantize fp8 \
  --quant-scales /path/to/scales.json \
  -o /path/to/model-fp8.bundle
```

The current parser accepts:

```text
fp8, int8, int8_sq, int4, int4_awq, nvfp4, w4a8
```

`int8` is normalized to `int8_sq`, and `int4` is normalized to `int4_awq`.
When no scale file is supplied, the framework selects calibration,
pre-quantized checkpoint extraction, or dynamic scales according to the
canonical format and checkpoint metadata. Control the PTQ calibration budget
with:

```bash
--quant-calibration-samples 512
```

### Why ModelOpt is a build dependency

TRTMC uses NVIDIA ModelOpt only when automatic PTQ calibration is selected
and no precomputed scale artifact is supplied. A family adapter loads the
reference model and representative batches, ModelOpt runs the format-specific
calibration algorithm, and TRTMC extracts the resulting scale state into its
own `QuantScaleMap`. TRTMC then builds the TensorRT graph and packages the
native bundle; ModelOpt is not a runtime backend and is not required merely to
load a bundle whose scales and engines are already present. This reuses the
calibration algorithms without transferring family selection, graph
construction, or runtime ownership out of TensorRT-Model-Connect.

Inspect the live parser rather than copying an old option list:

```bash
trtmc build --help
```

### Qwen-VL calibration inputs

When a Qwen2.5-VL or Qwen3-VL build enters calibration, its family adapter
loads the full vision-language model through
`AutoModelForImageTextToText` or `AutoModelForVision2Seq` and collects
image-plus-text activation statistics. Supply representative paired samples
with `TRTMC_CALIB_MANIFEST`; each non-empty JSONL row has this shape:

```json
{"image": "/absolute/path/example.jpg", "prompt": "Describe the image."}
```

```bash
export TRTMC_CALIB_MANIFEST=/absolute/path/qwen-vl-calibration.jsonl
```

If no readable paired manifest is set, `TRTMC_CALIB_IMAGE_DIR` selects up to
64 images and combines them with calibration prompts. If neither source is
available, the adapter uses synthetic placeholder images and says so in its
output. A processor failure for one image-plus-text batch falls back to
text-only tokenization for that batch.

Real calibration images should represent the deployment workload and remain
disjoint from evaluation data. The presence of this adapter does not qualify
every Qwen-VL model, precision, or quantization format; retain model-specific
graph, parity, output-health, and performance evidence.

## Ownership standard

The shared package under
`python/tensorrt_model_connect/quantization/` provides:

- `QuantPlan`, which records the canonical format, base precision, scale
  source, scale artifact, and calibration budget;
- `QuantContext`, which lets a family-owned graph route selected operations
  through quantized implementations;
- `QuantProfile` and `QuantScaleMap`, which decide which weights are
  quantized and carry their scales;
- format implementations and scale providers for precomputed, calibrated,
  pre-quantized, and dynamic paths; and
- reusable calibration-adapter primitives.

### Family agent default scope

Each family `model.py` remains responsible for:

- accepted formats and explicit rejection of unsupported combinations;
- graph locations where `QuantContext` is consumed;
- exclusions, calibration inputs, and pre-quantized checkpoint semantics;
- tensor-parallel compatibility; and
- parity, output-health, and performance qualification.

Family-local functions may include:

- `quant_exclude_patterns()`
- `calibration_data()`
- `quant_adapter()`
- `fp8_precomputed_scales()`
- `fp8_calibrate()`
- `supports_parallel_quantization()`

The cross-owner invariants are:

- Shared quantization code must not import specific family model modules.
- Shared quantization code must not branch on concrete family names.
- Family-specific quantization policy belongs in local functions such as
  `quant_adapter()` and `quant_exclude_patterns()`.

### Core agent scope

The core scope is the model-independent implementation under
`python/tensorrt_model_connect/quantization/`. A family change becomes a core
change only when it adds or fixes a shared primitive such as a format, scale
contract, common graph seam, or shared calibration/runtime behavior.

The selected `model.py` creates and passes `quant_ctx` only through graph paths
that support it. Tensor-parallel quantization is separately family-gated; for
example, Qwen currently opts in only for FP8. Do not infer tensor-parallel
support from the presence of both CLI options.

## Test enforcement

`tests/builder/test_quantization_ownership.py` checks that the shared core does
not import family modules or branch on family names and that model-specific
hooks remain family-owned. These static checks protect ownership boundaries;
they do not replace build, parity, output-health, or performance qualification.

## Direct FP8 scale path

Some diffusion and family-specific recipes use direct FP8 scale functions
instead of `QuantContext`:

```bash
trtmc build /path/to/model \
  --fp8 \
  -o /path/to/model-fp8.bundle
```

`--fp8` first asks the family model for packaged scales and then calls its
`fp8_calibrate()` function if no matching asset exists. A family without either
path fails and asks for an explicit scale file.

Load or save scales with:

```bash
--fp8-scales /path/to/scales.json
--save-fp8-scales /path/to/calibrated-scales.json
```

An explicit `--fp8-scales` input must be a readable UTF-8 JSON object. Missing
files, malformed JSON, and top-level arrays or scalar values fail before the
native build starts. One valid shape is:

```json
{
  "transformer.block.0": {
    "input_scale": 0.5,
    "weight_scale": 0.25
  }
}
```

Prefer `--quantize fp8` when the family graph consumes `QuantContext`. Use the
direct scale flags only when the owning family implements that FP8 contract.

## Build-path selection

`trtmc build` resolves the owning family and calls its `model.build()` once.
The family owns implementation selection. Qwen may match one exact qualified
optimized-runtime profile for the model revision, target, and public options;
no match runs Qwen's native recipe, while a selected provider failure is
terminal. Other families run their native recipes directly.

Precision, quantization, calibration, and scale options are part of that
selection tuple. Changing one can therefore change both engine numerics and
the implementation path.

After the build, inspect the bundle:

```bash
trtmc inspect /path/to/model-fp8.bundle
```

An `optimized_runtime.json` section identifies an optimized bundle. Native
bundles instead record their model-owned runtime strategy and quantization
metadata. Record the bundle kind with all comparison results.

## Qualification evidence

A quantized model claim should retain:

1. the exact model ID and immutable revision;
2. the format, base precision, scale source, and calibration data/version;
3. the build command, code revision, and generated bundle;
4. bundle inspection showing the selected implementation path;
5. a family- and task-appropriate reference comparison;
6. user-visible output-health evidence; and
7. hardware, warmups, repetitions, and raw artifacts for any performance
   claim.

Useful task-specific checks include logits and generated text for decoders,
image/video health and reference comparisons for diffusion, waveform or ASR
checks for audio, and numerical error metrics for time-series models. A generic
unit test of the shared format core is not model qualification.

{/* Collaborative review anchor: batch 2. */}
