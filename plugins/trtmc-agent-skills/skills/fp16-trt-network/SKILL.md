---
name: fp16-trt-network
description: >-
  Use when adding, reviewing, or debugging FP16/BF16 precision support in
  strongly typed TensorRT networks. Covers dtype threading, FP32 precision
  boundaries, TensorRT strongly typed restrictions, bundle metadata, and common
  FP16 failure modes.
---

# FP16 TensorRT Network Construction

## Core Rule

In strongly typed TensorRT networks, precision is controlled by tensor dtypes,
typed constants, and explicit `network.add_cast(...)` boundaries. Do not use
`BuilderFlag.FP16`, `BuilderFlag.INT8`, `layer.setPrecision()`,
`layer.setOutputType()`, or direct `tensor.dtype = ...` assignment.

```python
flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flags)
```

## Precision Boundary Map

Keep these operations in FP32:

| Operation | Reason |
|-----------|--------|
| RMSNorm, LayerNorm, GroupNorm, L2Norm | Reduction and reciprocal/sqrt precision |
| Softmax | FP16 exp overflow and probability normalization error |
| BatchNorm | Running mean/variance and affine arithmetic |
| Final logits output | Accurate token ranking and sampling |

These operations are usually safe in FP16/BF16 work dtype:

| Operation | Notes |
|-----------|-------|
| MatMul / linear projection | Tensor Cores use FP16 input with higher precision accumulation |
| Embedding lookup | Table lookup, no arithmetic |
| SiLU / GELU / ReLU | Elementwise in normal activation ranges |
| RoPE elementwise application | Cos/sin values fit in FP16 |
| Residual add and bias add | Safe when values remain in expected range |

## Standard Pattern

Every sensitive operation should enter FP32, compute in FP32, then cast back to
the work dtype when needed:

```python
def add_rms_norm(network, inp, hidden_size, gamma, eps_tensor, dtype=np.float32):
    need_cast = dtype != np.float32
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    # Compute mean/sqrt/reciprocal/scale in FP32.

    if need_cast:
        result = network.add_cast(result, _np_to_trt_dtype(dtype)).get_output(0)
    return result
```

Critical details:

- Cast every tensor used inside the FP32 computation, including epsilon tensors.
- Norm weights (`gamma`, `beta`) stay FP32, even when other weights are FP16.
- In strongly typed mode, all inputs to an elementwise op must have the same
  dtype.
- Constants must be created in their target dtype; TensorRT will not infer a
  cast for you.

## Builder Checklist

- Accept a precision parameter, usually `precision: str = "fp32"`.
- Compute both work dtypes:
  `work_np_dtype = np.float16 if precision == "fp16" else np.float32` and
  `work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32`.
- Set KV cache or recurrent state tensors to `work_trt_dtype`.
- Keep attention mask inputs FP32, then cast to the work dtype if an elementwise
  op requires it.
- Pass `dtype=work_np_dtype` to constants, graph ops, and graph blocks that use
  work-precision weights.
- Keep norm constants and norm affine parameters FP32 inside the boundary.
- Cast logits or final comparison-sensitive outputs to FP32 before
  `network.mark_output(...)`.
- Write `"precision": precision` into bundle metadata.
- Add or update an E2E manifest with `"precision": "fp16"` when adding a
  persistent FP16 variant.

## Weight Loading

| Weight type | FP16 mode dtype |
|-------------|-----------------|
| Embeddings, Q/K/V/O projections, MLP, LM head | `np.float16` |
| RMSNorm/LayerNorm gamma and beta | `np.float32` |
| Q/K norm gamma and per-head norm weights | `np.float32` |

## Common Failures

- Repeated tokens or degenerate text: missing FP32 norm boundary.
- TensorRT build error about mismatched elementwise types: constants or epsilon
  tensors were created in the wrong dtype.
- Masked tokens affect output: attention mask was converted to FP16 too early
  or used values outside FP16 range.
- Plausible but wrong text: logits or final comparison outputs stayed FP16.
- FP16 bundle is nearly the same size as FP32: the builder accepted the flag but
  did not actually thread the work dtype through weights and inputs.

## Validation

- Inspect the bundle with `trtmc-build inspect <bundle.trtfb>`.
- Compare FP32 and FP16 bundle sizes; FP16 weight-heavy bundles should be
  materially smaller.
- Run the narrow E2E manifest or a targeted diff command before claiming the
  precision change is correct.
- Use `$debug-trt-mismatch` if low-precision output diverges unexpectedly.
