---
title: Phi4 Multimodal with Edge-LLM
---

# Phi4 Multimodal with Edge-LLM

The `phi4_multimodal` family owns the optional Edge-LLM 0.11 complete-network
offload for `microsoft/Phi-4-multimodal-instruct`. Build and runtime selection
remain inside the family; no global execution-mode switch is required.

## Setup and use

Install Model Connect with its optional native Edge SDK and expose that SDK's
prefix through `CMAKE_PREFIX_PATH`. A compatible Edge 0.11 Python wheel can
supply the builder/export tools; the native C++ SDK is still required.
Build on the target GPU rather than cross-compiling.

For original checkpoints, use FP16 execution, tensor parallel size1, and batch1.
The adapter applies the vision LoRA specified by the official Phi4 workflow.
Already merged ModelOpt FP8, NVFP4, and W4A16-AWQ checkpoints use the family's
ONNX route with batch2 and context8192. Quantized checkpoints are inputs, not
automatically calibrated outputs of a Model Connect build.

The runtime accepts image-to-text and raw no-image prompts. It does not expose
audio or claim a separate unmerged Hugging Face LANGUAGE-mode engine.

## Validation limitations

The original image test and its accuracy threshold are unchanged. The
context8192 FP16 output meets the text-quality gate; smaller-context and
quantized configurations have recorded accuracy failures and remain runnable.
A working build and inference call is not a claim of quality qualification.
Audio remains blocked by the pinned builder component and task interfaces.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/phi4_multimodal/edge_llm/README.md)
for exact route, precision, profile, and failure-handling boundaries.
