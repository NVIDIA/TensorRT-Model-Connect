---
title: Nemotron-H with Edge-LLM
---

Nemotron-H owns its optional Edge-LLM 0.11.0 builder, tokenizer contract, and
C++ runtime adapter. The ordinary family command selects Edge when a matching
native SDK and supported configuration are available.

```sh
trtmc nemotron_h build /models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --precision fp16 --max-sequence-length 256 --output nemotron.bundle
trtmc nemotron_h build /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --precision fp16 --max-sequence-length 256 \
  --execution-variant dspark --companion draft=/models/lightning-dspark \
  --output lightning-dspark.bundle
```

The listed Lightning companion modes are `dflash`, `dspark`, and
`dspark_tree`. DFlash and DSpark tree require greedy generation; DSpark
chain preserves supported sampling controls. Paired execution builds both
engines using the ONNX flow and never substitutes a base-only bundle.

Nine exact local profiles passed: 4B BF16/FP8, 9B BF16/FP8/NVFP4, Lightning
NVFP4, and its three companion modes. Nano30B NVFP4 builds and runs but has a
recorded output-budget accuracy failure; it remains runnable. Super120B NVFP4
was not executed because compatible single-device capacity was insufficient.
These are bounded text-only results, not qualification of all contexts or
platforms. Nemotron Omni and ASR are separate interfaces.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_h/edge_llm/README.md)
for exact profiles, unchanged quality gates, source revisions, and limitations.
A compatible pip installation may provide Python tools; native C++ inference
still requires the matching provisioned SDK. Ordinary preparation failures
warn and attempt native build once; runtime failures propagate.
