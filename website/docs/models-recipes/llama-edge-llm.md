---
title: Llama with Edge-LLM
---

Llama owns its optional Edge-LLM 0.11.0 builder and C++ runtime adapter.
Use the normal family build command; a matching native SDK and supported
configuration select Edge without a shared backend selector.

```sh
trtmc llama build /path/to/Llama-3.1-8B-Instruct --precision fp16 -o llama.bundle
trtmc llama build /path/to/Llama-3.1-8B-Instruct --precision fp16 \
  --execution-variant eagle3 --companion draft=/path/to/EAGLE3-draft \
  -o llama-eagle3.bundle
```

Original 3.1-8B and 3.2-3B FP16 profiles passed local text validation on SM80.
The 3.1-8B EAGLE3 pair and NVIDIA packed FP8/NVFP4 8B profiles passed on SM120.
These results cover the exact capacities and greedy cases in the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/llama/edge_llm/README.md),
not every configuration or decoding policy.

Llama 3.2-1B remains executable but has a recorded semantic accuracy failure.
The access-blocked Llama 3.0 checkpoint has no new Edge profile.
A compatible pip installation can supply Python builder tools; the native SDK
is still required for C++ inference. Ordinary preparation failures warn and
attempt the native builder; an explicit companion failure never produces a
base-only bundle.

## Native fallback

An unavailable Edge route uses the owning TensorRT Python graph and C++ runtime,
without ONNX conversion or a native GPU allowlist. Explicit EAGLE3 requests keep
both the target and companion; the native verifier supports the family sampler
and seeded requests without claiming Edge's tree schedule or a speedup.

Prepared FP8/NVFP4 sources preserve packed weights and calibrated scales. The
automatic route needs no new calibration or Torch weight reader. FP8 cache
values are rounded through Q/DQ but remain in FP16 buffers, so this route does
not claim packed-cache memory savings. The existing explicit quantization
options remain available.

Retained native evidence covers the original EAGLE3 pair on SM80, prepared FP8
on SM89 and prepared NVFP4 on SM120. A 20GB SM89 NVFP4 context OOM and the
TensorRT 11.1 SM80 FP8-operator limitation remain documented failures, not
adapter SKU exclusions. See the family recipe for exact revisions, capacities,
quality gates and sampled same-target comparisons. These historical SDK results
do not qualify every platform or the current repository-pinned SDK.
