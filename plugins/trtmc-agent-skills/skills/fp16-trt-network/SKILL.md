---
name: fp16-trt-network
description: >-
  Implement or review FP16 and BF16 behavior in one family-owned, strongly
  typed TensorRT network without changing unrelated families.
---

# FP16 and BF16 TensorRT Networks

Work in the owning `families/<family>/` implementation. Read its `model.py`,
local graph helpers, manifests, and tests before changing dtype behavior.

## Preserve the typed graph

- Create the network with
  `trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED`.
- Resolve `request.precision` explicitly and reject unsupported values.
- Track checkpoint storage dtype, TensorRT tensor dtype, shape, and layout at
  every cast-sensitive boundary.
- Create constants in an intentional storage dtype and use
  `network.add_cast(...)` when the runtime dtype differs.
- Keep all inputs to an elementwise operation dtype-compatible, including
  epsilon, masks, scales, and affine parameters.
- Preserve compact K/V projection and cache shapes for grouped or multi-query
  attention.

Use FP32 only where the family implementation or measured reference evidence
requires it, commonly reductions, normalization, probability arithmetic, or
comparison-sensitive outputs. Cast back explicitly. Do not apply a blanket
FP32 layer list or remove an established family boundary without evidence.

Precision and quantization are separate request values. Change one at a time,
and do not add a shared model helper merely because another family has similar
code. Copy a small proven pattern into the owner when needed.

## Prove the change

Build through the public Python build command with the exact checkpoint and
precision. Inspect the resulting bundle, run focused family Python/C++ tests,
then run the declared family E2E testcase against its existing oracle.

Bundle size is only a signal. Qualification requires the requested graph path
to execute and the unchanged family correctness contract to pass. Report any
hardware or precision variant not run.
