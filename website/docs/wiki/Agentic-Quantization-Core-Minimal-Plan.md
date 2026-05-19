# Agentic Quantization Core: Minimal Plan

Status: implemented v1 slice

This document describes the minimum quantization architecture needed for
agent-driven rollout without coupling model builders to each other.

## Shared Core

The shared core is intentionally small:

1. `QuantPlan`
   - canonical description of one build attempt
   - normalizes format aliases like `int8 -> int8_sq`
   - records `base_precision`, `quant_format`, `scale_source`,
     `scale_artifact`, and calibration budget

2. `ScaleMap`
   - canonical per-op scale storage
   - still keyed by builder seam names in the current standard decoder path

3. `FormatKernel`
   - shared QDQ emitters in `tensorrt_model_connect/tensorrt_model_connect/quantization/formats.py`
   - owns format math, not model semantics

4. `ScaleProvider`
   - shared scale acquisition layer
   - v1 uses:
     - `PrecomputedJsonProvider`
     - `ModelOptCalibrationProvider`

5. `Validation Contract`
   - E2E manifests now accept a generic `quantization` block
   - orchestrator converts that block into `./build/trtmc build` args

Authoritative shared-core files for quantization:

- `tensorrt_model_connect/tensorrt_model_connect/quantization/plan.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/context.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/formats.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/profile.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/scales.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/scale_providers.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/adapters.py`
- `tensorrt_model_connect/tensorrt_model_connect/quantization/__init__.py`
- `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py`
- `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py`
- `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py`

## Family-Local Layer

Family ownership stays local:

1. `FamilyQuantAdapter`
   - bridges reference model loading, calibration inputs, and scale-name
     mapping
   - provided via `plugin.quant_adapter(format_name)` when a family needs
     custom calibration behavior

2. `FamilyQuantSeam`
   - the family builder decides where quantization is allowed
   - the shared format layer decides how QDQ is emitted

For the current v1 slice, the standard decoder path already had the seam
through `quant_ctx`, so only the adapter boundary had to be added.

## Ownership Standard

This section is normative. It defines the boundary that parallel agents are
expected to follow.

1. Family agent default scope
   - A family agent defaults to family-local files only.
   - Examples:
     - `tensorrt_model_connect/tensorrt_model_connect/families/<family>.py`
     - family-specific builder files
     - family-specific manifests and tests

2. Core agent scope
   - A core agent owns the shared-core files listed above.
   - Family agents should not edit those files during normal model onboarding.

3. Escalation rule
   - A task becomes a core task only when it adds or fixes a shared primitive:
     - new quantization format
     - new shared op seam
     - shared scale contract change
     - shared dtype/runtime/reference bug

4. Shared-core hygiene
   - Shared quantization code must not import specific family plugins.
   - Shared quantization code must not branch on concrete family names.
   - Family-specific quantization policy belongs in plugin hooks such as
     `quant_adapter()` and `quant_exclude_patterns()`.

5. Review rule
   - If a family rollout appears to require a shared-core edit, that edit
     should be isolated and reviewed as a core delta, not hidden inside the
     family change.

## What Was Implemented

Code changes in this branch:

- added `tensorrt_model_connect/tensorrt_model_connect/quantization/plan.py`
- added `tensorrt_model_connect/tensorrt_model_connect/quantization/adapters.py`
- refactored `ModelOptCalibrationProvider` to use a calibration adapter
  instead of hardcoding `AutoModelForCausalLM` logic directly
- taught `build_quant_context()` to consume a `QuantPlan`
- normalized CLI quantization aliases
- taught the E2E manifest/orchestrator path to consume a generic
  `quantization` block
- added a qwen FP8 E2E manifest

## Validation Result

Validated locally with `Qwen/Qwen3-0.6B`.

Working path:

- `FP8 + FP16 base`
- ModelOpt calibration through the new default decoder adapter
- bundle build succeeded
- bundle inspect showed:
  - baseline bundle: `3.1G`
  - quantized bundle: `1.6G`
  - precision: `fp16`
  - quantization: `fp8`
- direct runtime generation produced non-empty text
- E2E case `qwen3-0.6b-fp8` passed end to end

Known limitation discovered during validation:

- `FP8 + BF16 base` is still broken in the existing standard decoder builder
  path because some tensors remain `Half` while cache tensors are `BFloat16`
  at concatenation boundaries
- this is a builder precision bug, not a quantization contract bug

## Why This Is The Minimum Viable Shape

This split is small enough to scale:

- adding a new family should only require a family-local adapter and local
  builder seam work
- adding a new format should only require shared format-kernel work
- adding a new scale source should only require provider work

This avoids the two main failure modes:

1. a global quant context that every family must edit
2. format math duplicated independently inside every builder

It also keeps parallel rollout sane:

- most new families should land without touching shared core
- shared core stays small enough for a small number of core agents to own
- family-local work remains embarrassingly parallel

## Test Enforcement

The repo should enforce this standard in CI:

- shared quantization core files must exist as a small fixed set
- shared quantization core must not import concrete family modules
- shared quantization core must not contain family-name branches
- family-specific quantization hooks must live in the family plugin

## Next Steps

Priority order:

1. fix BF16 threading in the standard decoder path so `FP8 + BF16 base`
   works as intended
2. add a diffusion family adapter and manifest using the same `QuantPlan`
   and `quantization` manifest contract
3. wire pre-quantized checkpoint extraction into the main provider path
4. extend coverage reporting so agents can detect "built but silently mostly
   fallbacked" quantized bundles
