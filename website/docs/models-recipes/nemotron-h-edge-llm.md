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
chain preserves supported sampling controls. Edge-backed paired execution builds
both engines using the ONNX flow and never substitutes a base-only bundle.

Nine exact Edge-backed profiles passed: 4B BF16/FP8, 9B BF16/FP8/NVFP4, Lightning
NVFP4, and its three companion modes. Nano30B NVFP4 builds and runs but has a
recorded output-budget accuracy failure; it remains runnable. Super120B NVFP4
was not executed because compatible single-device capacity was insufficient.
These are bounded text-only results, not qualification of all contexts or
platforms. Nemotron Omni and ASR are separate interfaces.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_h/edge_llm/README.md)
for exact profiles, unchanged quality gates, source revisions, and limitations.
A compatible pip installation may provide Python tools; Edge-backed C++ inference
still requires the matching provisioned SDK. Preparation failures warn and attempt
the unchanged native request once; runtime failures propagate.

## Native fallback

Without compatible Edge offload, the family uses TensorRT Python graph construction
and the Model Connect C++ runtime, without ONNX. Prepared FP8, NVFP4 and mixed
source formats, scales and FP8 KV storage are preserved; no replacement checkpoint
is calibrated. A paired request still builds the target and actual companion,
including the DSpark Markov head, rather than silently using ordinary decoding.

The exact native FP16-compute, TP1/batch1, capacity-256 results are separate from
the Edge results above: seven profiles pass, while Nano30B, ordinary Lightning
and Lightning DSpark chain retain accuracy failures and remain runnable.
Super120B has a demonstrated TensorRT compiler stack-exhaustion failure on the
tested ARM configuration and no native inference qualification. The larger-stack
diagnostic was stopped after more than 14 hours; this is not an observed Edge
failure or proof of a compiler hang. See the family recipe for individual scores,
speculative target-parity checks and the unresolved synthetic draft diagnostic.
These results do not qualify every platform or capacity, and tested GPU types
are not native family admission restrictions.
