---
name: fp16-trt-network
description: >-
  Use when adding, reviewing, or debugging FP16/BF16 precision in a
  family-owned, strongly typed TensorRT network. Covers dtype threading,
  explicit FP32 boundaries, typed constants, compact GQA/MQA state, bundle
  evidence, and low-precision validation.
---

# FP16/BF16 TensorRT Networks

## Contract

TensorRT networks in this repository are strongly typed:

```python
from tensorrt_model_connect import trt_compat

flags = trt_compat.network_creation_flags(strongly_typed=True)
network = builder.create_network(flags)
```

Follow an owning family's established direct flag when it is intentionally
backend-specific. For new backend-agnostic code, use `trt_compat` so TensorRT
versions without `EXPLICIT_BATCH` and the optional TensorRT-RTX backend share
one flag boundary. Backend selection, including `--rtx`, must happen before a
module imports TensorRT; never switch backends after TensorRT is loaded.

Precision follows tensor dtypes, typed constants, and explicit
`network.add_cast(...)` boundaries. Do not use `BuilderFlag.FP16`,
`BuilderFlag.INT8`, `layer.setPrecision()`, `layer.setOutputType()`, or direct
`tensor.dtype` mutation to override inference in a strongly typed network.

Keep changes in the owning family under
`python/tensorrt_model_connect/models/<family>/`. Root graph helper modules
are intentionally absent. Share a helper only within an ownership boundary
where shape, dtype, and layout semantics genuinely match.

## Map Precision Before Editing

For every input, weight, constant, intermediate, state tensor, and output,
record:

- storage dtype used to create the constant;
- TensorRT runtime dtype;
- shape and layout;
- the operation where a cast occurs;
- the required comparison dtype.

BF16 needs special care. Some family builders store constants in FP16-compatible
NumPy storage and explicitly cast them to `trt.bfloat16`. Do not assume a
NumPy dtype maps directly to the TensorRT dtype. Follow the owning family's
constant helper and checkpoint mapper.

## FP32 Boundaries

Use the family implementation and reference numerics to decide boundaries.
Common FP32 candidates include:

- normalization reductions and reciprocal/square-root arithmetic;
- softmax and probability normalization;
- batch/group statistics;
- final logits or comparison-sensitive outputs;
- unstable scale or calibration arithmetic.

Matrix multiplies, projections, embeddings, activations, RoPE, residuals, and
biases often use the work dtype, but this is not a universal license. Preserve
an established family-specific FP32 layer or operation.

The boundary pattern is:

```python
fp32_in = network.add_cast(inp, trt.float32).get_output(0)
fp32_eps = network.add_cast(eps_tensor, trt.float32).get_output(0)
# Compute the sensitive operation with FP32 inputs and FP32 constants.
result = network.add_cast(fp32_result, work_trt_dtype).get_output(0)
```

Rules:

- all inputs to a strongly typed elementwise operation must agree in dtype;
- create constants in the intended storage dtype and cast when runtime dtype
  differs;
- include epsilon, masks, scale tensors, and affine parameters in the map;
- cast marked outputs to the comparison contract's dtype;
- do not replace finite masking with values outside the low-precision range.

## GQA/MQA State

TensorRT attention supports grouped- and multi-query attention. Keep K/V
projection weights, biases, and cache tensors at compact
`num_key_value_heads * head_dim` width unless a specific non-attention
operation proves a different layout is required. Do not expand K/V to query
head width merely to simplify graph construction.

Verify:

- Q, K, and V shapes before attention;
- cache allocation and update shapes;
- tensor-parallel partitioning constraints;
- bundle/runtime metadata describing cache layout;
- Python debug runner and C++ runtime agreement.

## Quantization Interaction

Low-precision base dtype and quantization are separate effective build options.
Changing either may select a different optimized implementation instead of the
native builder. Follow the family quantization hooks and shared quantization
plan; do not add ad hoc Q/DQ logic to a generic precision helper.

Record whether each compared bundle is native or optimized. A result that
changes both precision and runtime implementation does not isolate precision.
Likewise, compare TensorRT and TensorRT-RTX separately: a backend change is not
evidence for a dtype-only change.

## Implementation Checklist

- Resolve `fp32`, `fp16`, and `bf16` explicitly; fail closed on unsupported
  values.
- Thread both storage and TensorRT work dtypes through the complete graph.
- Preserve strongly typed network creation on every builder path.
- Preserve the selected TensorRT versus TensorRT-RTX backend and reject a
  precision/backend combination the owning family does not support.
- Keep compact K/V shapes for GQA/MQA.
- Keep family-specific graph logic family-owned.
- Ensure bundle config records the requested precision and selected runtime
  path.
- Add or update a model-owned manifest/profile instead of inventing a
  standalone example.

## Failure Signatures

| Symptom | Inspect first |
|---|---|
| Build-time elementwise dtype error | constant, epsilon, mask, or BF16 cast |
| Repeated/degenerate tokens | norm, softmax, cache, or logits boundary |
| First decode step differs | weight storage, prefill dtype, or config |
| Divergence grows by token/layer | accumulation or missing FP32 boundary |
| FP16 bundle is not materially smaller | actual section/weight dtypes and runtime path |
| BF16 passes FP16 but fails BF16 | storage-to-runtime casts and unsupported ops |

## Validation

Inspect the artifact and selected path:

```bash
./build/trtmc inspect <bundle.bundle>
```

Bundle size is only a signal. Confirm section metadata and weight dtypes; do
not declare success from an approximate 2x size change.

Run the owning model-first validation or focused E2E case with fixed model
revision, workload, sampling, and runtime strategy:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py <model> <workload> \
  --bundle <bundle.bundle> \
  --output <artifact-dir>
```

Compare FP32 and low-precision artifacts with the same implementation path.
Use `$debug-trt-mismatch` for the first divergent token, layer, stage, or
runtime boundary. Report checks that were not run on target hardware.

<!-- Collaborative review anchor: batch 2. -->
