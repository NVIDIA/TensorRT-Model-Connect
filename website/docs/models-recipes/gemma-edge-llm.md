---
title: Gemma4 with Edge-LLM
---

Gemma owns its optional Edge-LLM 0.11.0 ONNX builder and C++ runtime adapter.
The standalone path handles text and the checkpoint's image/audio towers;
explicit companions select MTP, DSpark, EAGLE3 or DFlash text execution.

```sh
trtmc gemma build /path/to/gemma-4-E2B-it --precision fp16 -o gemma.bundle
trtmc gemma build /path/to/gemma-4-12B-it --precision fp16 \
  --execution-variant mtp --companion draft=/path/to/assistant -o gemma-mtp.bundle
```

Provision the optional native Edge SDK with ONNX tools and all native kernels.
Python installation alone does not provide the native SDK. All selection and
media mapping stay inside Gemma; earlier generations retain native behavior.

When Edge is unavailable or its builder fails, the family-owned native fallback
constructs TensorRT graphs directly in Python and executes them through the
Gemma C++ runtime, without ONNX. It preserves the requested checkpoint format
and companion rather than silently substituting ordinary decoding.
Native FP8 attention keeps calibrated projection storage and uses explicit
FP32 score, mask and softmax math, without a GPU-specific dispatch rule.

Recorded native cases include E2B/E4B/12B and the original or published prepared
26B/31B variants. Sampled 12B DSpark, EAGLE3 and both DFlash draft architectures
have actual draft-execution and same-native-target parity evidence. This is
separate from Edge's sampled EAGLE3/DFlash vanilla-fallback limitation and does
not claim a speculative speedup or cross-framework random-number equivalence.

Command-successful accuracy failures remain executable. Consult the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md)
for exact checkpoints, tested profiles, historical versus current-source MTP
evidence, accuracy failures and installation requirements. These results do
not qualify every modality, capacity, generation policy or platform.
