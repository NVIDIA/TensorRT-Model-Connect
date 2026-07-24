# Quantization Core Implementation Note

:::info Implementation snapshot

This page replaces the earlier forward-looking plan with the current shared
contract. Actual format support remains family-dependent and must be proved by
the selected family's build and E2E evidence.

:::

## Shared core

`python/tensorrt_model_connect/quantization/` owns:

- `QuantPlan`: normalized requested format, scale source, calibration budget,
  and exclusions
- `QuantContext`: build-time quantization state passed to a compatible family
  builder
- `ScaleMap` and scale providers
- format adapters/emitters
- registry and profile helpers

The build CLI accepts:

```bash
PYTHONPATH=python python3 -m tensorrt_model_connect build --help
```

The parser currently exposes `--quantize` values `fp8`, `int8`, `int8_sq`,
`int4`, `int4_awq`, `nvfp4`, and `w4a8`, plus scale and calibration options.
Parser acceptance is not a claim that every family supports every format.

## Ownership rule

The shared core owns format mechanics. A family owns:

- which formats it accepts
- model-specific exclusions and calibration adapters
- graph seams where quantization is applied
- pre-quantized checkpoint interpretation
- tensor-parallel compatibility
- parity, health, and performance evidence

The engine builder passes a quantization context only to plugins whose build
entry point accepts it. That signature check is not a global support gate: a
plugin can currently accept and ignore the context. Therefore parser
acceptance and even a successful build do not prove that quantization was
applied. Each family should reject unsupported combinations explicitly, and
qualification must verify the resulting bundle metadata and task output.

## Evidence

For a quantized model, retain the exact model revision, format, scale source,
calibration data/version, build command, bundle, comparison artifact, hardware,
and performance artifact when performance is claimed. A generic unit test of
the format core is not model qualification.
