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
