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

The E2B, E4B and 12B rollout has passing modes and recorded accuracy failures.
Command-successful accuracy failures remain executable; sampled EAGLE3/DFlash
fallback is not claimed as successful speculative decoding. Consult the
[family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/gemma/edge_llm/README.md)
for exact tested modes, limitations and installation requirements. These results
do not qualify larger checkpoints or every modality/configuration combination.
