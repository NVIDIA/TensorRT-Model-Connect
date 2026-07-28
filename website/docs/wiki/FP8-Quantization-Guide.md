# FP8 Quantization

FP8 support is family-dependent. Use the public build CLI and the selected
family's quantization adapter; do not copy old root-level graph-helper recipes.

## Build with the generic quantization path

```bash
PYTHONPATH=python python3 -m tensorrt_model_connect build \
  /path/to/model \
  --quantize fp8 \
  --quant-scales /path/to/scales.json \
  -o /path/to/model-fp8.trtfb
```

If the family supports calibration through the generic quantization context,
omit `--quant-scales` and set the calibration budget:

```bash
PYTHONPATH=python python3 -m tensorrt_model_connect build \
  /path/to/model \
  --quantize fp8 \
  --quant-calibration-samples 512 \
  -o /path/to/model-fp8.trtfb
```

The legacy family hook is also exposed as `--fp8`, `--fp8-scales`, and
`--save-fp8-scales`. Prefer `--quantize fp8` for families integrated with
`QuantContext`; use the legacy flags only when the owning family implements
its `fp8_calibrate`/FP8-scale path. An explicit `--fp8-scales` file must be
readable UTF-8 JSON with an object at the top level. For example:

```json
{
  "transformer.block.0": {
    "input_scale": 0.5,
    "weight_scale": 0.25
  }
}
```

The CLI rejects a missing or unreadable file, malformed JSON, and arrays or
scalar values before starting the native build. Inspect the live parser with:

```bash
PYTHONPATH=python python3 -m tensorrt_model_connect build --help
```

## Current code boundaries

- Shared format and scale mechanics:
  `python/tensorrt_model_connect/quantization/`
- Build orchestration:
  `python/tensorrt_model_connect/engine_builder.py`
- Family-specific support:
  `python/tensorrt_model_connect/families/<family>/`
- Model-specific manifests and thresholds:
  `tests/e2e/models/<family>/`

The former root-level graph-operations module has been retired; graph semantics
live with their owning families.

## Validation

FP8 completion requires:

1. a build that records FP8 quantization metadata;
2. a family-appropriate reference comparison;
3. output-health validation for the user-visible task;
4. focused unit/build tests for the quantization seam; and
5. exact-hardware performance evidence before claiming a speedup.

Tensor-parallel quantization is explicitly family-gated. Do not assume it is
supported because both options appear in CLI help.
