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

## Ownership standard

This section is normative. It defines the boundary for parallel model and core
work.

The shared core owns format mechanics. A family owns:

- which formats it accepts
- model-specific exclusions and calibration adapters
- graph seams where quantization is applied
- pre-quantized checkpoint interpretation
- tensor-parallel compatibility
- parity, health, and performance evidence

### Family agent default scope

A family agent default scope is family-owned code and evidence:

- `python/tensorrt_model_connect/families/<family>/`, including `plugin.py`
  and family-owned builders
- family manifests, comparison policy, and tests

Family work should use shared quantization interfaces without adding
family-specific behavior to the shared core.

### Core agent scope

A core agent scope is the shared implementation under
`python/tensorrt_model_connect/quantization/`.

### Escalation rule

A family task becomes a core task only when it adds or fixes a shared primitive
such as a format, scale contract, common graph seam, or shared
calibration/runtime behavior.

### Shared-core hygiene

- Shared quantization code must not import specific family plugins.
- Shared quantization code must not branch on concrete family names.
- Family-specific quantization policy belongs in plugin hooks such as
  `quant_adapter()` and `quant_exclude_patterns()`.

### Review rule

If family onboarding reveals a required core change, isolate that change for
core review instead of hiding it inside the family delta.

The engine builder passes a quantization context only to plugins whose build
entry point accepts it. That signature check is not a global support gate: a
plugin can currently accept and ignore the context. Therefore parser
acceptance and even a successful build do not prove that quantization was
applied. Each family should reject unsupported combinations explicitly, and
qualification must verify the resulting bundle metadata and task output.

## Test enforcement

`tests/builder/test_quantization_ownership.py` enforces that:

- the fixed shared-core files exist
- shared quantization code does not import concrete family modules
- shared quantization code does not branch on concrete family names
- family-specific Qwen hooks remain in the Qwen plugin

These static ownership checks protect the architecture boundary. They do not
replace family-specific build, parity, health, or performance qualification.

## Evidence

For a quantized model, retain the exact model revision, format, scale source,
calibration data/version, build command, bundle, comparison artifact, hardware,
and performance artifact when performance is claimed. A generic unit test of
the format core is not model qualification.
