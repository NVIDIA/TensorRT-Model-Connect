---
title: Llama with Edge-LLM
---

Llama owns its optional Edge-LLM builder and C++ runtime adapter. Edge-enabled
builds default to 0.11.0 during migration. To use the 0.12.0 adapter, select
that SDK when [building Model Connect](../getting-started/source-build.md)
and pass `--edge-builder-python /absolute/path/to/python` to the family build
command. This selects the matching Python builder, installed from PyPI or from
source.
Use the normal family build command; a matching native SDK and supported
configuration select Edge without a shared backend selector.

An installed Model Connect wheel uses its own native Edge SDK, not a
different SDK on `CMAKE_PREFIX_PATH`. To switch releases, use a matching
wheel or a standalone Model Connect builder and runner together.

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

On 0.12, both the original Llama 3.2-1B and its RedHatAI static FP8
checkpoint remain executable but miss the arithmetic quality case.
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
automatic route needs no new calibration or Torch weight reader. Prepared
checkpoints with calibrated FP8 K/V now retain one-byte E4M3 cache bindings
on native TensorRT. Older bundles retain their original cache storage. Use an
explicit matching quantization option or allow detection from the source.

Retained native evidence covers the original EAGLE3 pair on SM80, prepared FP8
on SM89 and prepared NVFP4 on SM120. A 20GB SM89 NVFP4 context OOM and the
TensorRT 11.1 SM80 FP8-operator limitation remain documented failures, not
adapter SKU exclusions. See the family recipe for exact revisions, capacities,
quality gates and sampled same-target comparisons. These historical SDK results
do not qualify every platform or the current repository-pinned SDK.
