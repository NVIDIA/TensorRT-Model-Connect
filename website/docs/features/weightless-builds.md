---
title: Weightless Builds
description: Build small TensorRT plans without loading large weights, then refit them from a self-contained bundle.
---

Weightless builds reduce TensorRT build memory by replacing large learned
constants with named null placeholders. The plans omit those weights; Model
Connect stores their real values **once in the same bundle** and refits each
engine when the bundle loads. Qwen3 is the first supported family.

## Use Qwen3

Use a [prepared source environment](../getting-started/source-build.md) with
TensorRT **11.3 or later**. The initial path supports dense Qwen3 text
continuation, the `trt` backend, FP16 or BF16, and one GPU. Quantization,
tensor/context parallelism, embedding, and graph transforms are rejected.
The option belongs to `trtmc qwen build`; other families and the shared build
command do not expose it.

```bash
# Build a self-contained bundle from the pinned showcase checkpoint.
trtmc qwen build Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --task text_continuation \
  --precision bf16 --backend trt --strip-weights \
  --max-sequence-length 1024 -o qwen3-weightless.bundle

trtmc inspect qwen3-weightless.bundle
trtmc run qwen3-weightless.bundle --runtime-root ./build-weightless \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 --temperature 0 --top-k 1
```

Set `--runtime-root` to your native library directory. The Python entry point is `families.qwen.cli.build_bundle` with
`families.qwen.build_request.BuildRequest(..., strip_weights=True)`. Omit the flag for an ordinary build.
The resulting bundle can move independently of the checkpoint directory;
loading uses no checkpoint paths or weight-location environment variables.

## Design and ownership

```text
checkpoint headers -> named placeholders -> stripped prefill/decode plans
checkpoint bytes   -> streamed shared weight section
                      |
                      v
bundle -> mapped weight views -> refit each engine -> contexts -> inference
```

| Owner | Responsibility |
| --- | --- |
| Qwen family | Checkpoint mapping, storage dtype/layout, placeholder selection, section schema, engine bindings, and correctness tests. |
| Qwen build entry | Explicit mode checks and atomic publication through `BundleWriter`. |
| Bundle core | Bounded, immutable section mapping and its lifetime. |
| TensorRT backend | Validate the complete engine weight inventory, refit synchronously, then create execution contexts. |

Qwen retains checkpoint storage types and matrix orientation. TensorRT graph
casts and matrix transpose operations express the computation in the requested
precision. Large named weights become placeholders; small constants such as
norms remain in each plan. TensorRT owns fusion, tactics and GPU lowering.

The bundle contains `engine.plan`, `prefill.plan`, `refit_weights`, `refit.json`,
runtime configuration (including an explicit stripped-plan marker), and tokenizer assets. `refit.json` version 1 describes
payload-relative byte ranges, tensor shapes/types, and the names required by
each plan. Both plans share the packaged values; tied source weights can also
share a byte range. The schema belongs to Qwen, not core.

Packaging copies tensor ranges in bounded chunks instead of assembling a full
weight blob in RAM. Loading maps the payload read-only, keeps it alive through
both refits, and releases it after module creation. Missing bindings, invalid
geometry, out-of-range bytes and unsupported schema versions fail loading.

## Limits and validation

“Weightless” describes the **plans**, not a weight-free deployment. The bundle
still carries one copy of the weights. Initialization reads and transfers those
values; separate engines still own their TensorRT device allocations. Refit
adds load-time work. This feature is neither weight streaming nor quantization,
and it does not relax TensorRT engine compatibility requirements.

The pinned BF16 and FP16 Qwen3-0.6B showcase cases use the family's existing
Hugging Face reference oracle. Packaging tests cover mixed lazy/materialized
weights and relocation; native tests cover missing weights, invalid metadata,
buffer lifetime, and exact engine names/types/counts. Other checkpoints and
platforms require their own validation; this example establishes no performance
guarantee.

See [NVIDIA's refit and weight-placeholder contract](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/refitting-engines.html)
for the underlying TensorRT requirements.
