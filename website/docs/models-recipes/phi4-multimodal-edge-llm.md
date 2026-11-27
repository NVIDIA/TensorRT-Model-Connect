---
title: Phi4 Multimodal with Edge-LLM
---

# Phi4 Multimodal with Edge-LLM

An installed Model Connect wheel uses its own native Edge SDK, not a
different SDK on `CMAKE_PREFIX_PATH`. To switch releases, use a matching
wheel or a standalone Model Connect builder and runner together.

## Edge 0.12

Build MC with `TRTMC_ENABLE_EDGELLM=ON` and `TRTMC_EDGELLM_RELEASE=0.12.0`;
this selects a matching native SDK at build time rather than switching ABI at
runtime. Install the matching Python builder from the official wheel or source.
For `microsoft/Phi-4-multimodal-instruct`, the family command is:

```bash
trtmc phi4_multimodal build "$MODEL_DIR" -o phi4.bundle --precision fp16 \
  --max-sequence-length 10368 --edge-builder-python "$EDGE_PYTHON"
trtmc run phi4.bundle --image "$IMAGE" --prompt "Describe this image." \
  --max-new-tokens 30 --temperature 1 --top-k 1
```

Both Edge and the warned native fallback passed the actual visual-feature
check and the unchanged image-answer quality gate on SM80 (NED 0.1241 and
0, respectively). A separate raw text-only completion executed on both but
missed the FP32 text-distance gate. The optional Edge route does not change
the native fallback's requested precision or context. The released direct
builder does not offer a Phi4 audio component; this owner does not expose
audio-conditioned generation. See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/phi4_multimodal/edge_llm/README.md)
for the exact scope of the results.

## Explicit Edge 0.11: historical profiles

The `phi4_multimodal` family owns the optional Edge-LLM 0.11 complete-network
offload for `microsoft/Phi-4-multimodal-instruct`. Build and runtime selection
remain inside the family; no global execution-mode switch is required.

## Setup and use

Install Model Connect with its optional native Edge SDK and expose that SDK's
prefix through `CMAKE_PREFIX_PATH`. A compatible Edge 0.11 Python wheel can
supply the builder/export tools; the native C++ SDK is still required.
Build on the target GPU rather than cross-compiling.

For original checkpoints, use FP16 execution, tensor parallel size 1, batch 1,
and an explicit `--max-sequence-length` of at least 768. Shorter or unspecified
contexts select the native path rather than Edge.
The adapter applies the vision LoRA specified by the official Phi4 workflow.
Already merged ModelOpt FP8, NVFP4, and W4A16-AWQ checkpoints use the family's
ONNX route with batch 2 and context 8192. Quantized checkpoints are inputs, not
automatically calibrated outputs of a Model Connect build.

When Edge is unavailable or its builder fails, the family native graph can
read those same prepared formats without ONNX. It preserves quantization scales
and checkpoint EOS tokens, but still exposes single-request execution rather
than a batched-generation API. Full native checks of FP8, both NVFP4 LM-head
policies, and W4A16-AWQ pass build, inference, and actual image-feature health
on the tested SM120 configuration. All four retain text-accuracy failures under
the unchanged NED gate and remain runnable; see the family-owned recipe below
for exact parameters and outcomes. These historical results predate the current Dynamic-HD/LongRoPE integration
and do not qualify that combined implementation, other platforms or capacities.

The runtime accepts image-to-text and raw no-image prompts. It does not expose
audio or claim a separate unmerged Hugging Face LANGUAGE-mode engine.

## Validation limitations

The original image test and its accuracy threshold are unchanged. The
context 8192 FP16 output meets the text-quality gate; smaller-context and
quantized configurations have recorded accuracy failures and remain runnable.
A working build and inference call is not a claim of quality qualification.
Audio remains blocked by the pinned builder component and task interfaces.

See the [family-owned recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/phi4_multimodal/edge_llm/README.md)
for exact route, precision, profile, and failure-handling boundaries.

The original FP16 checkpoint passes the existing native image test at context
768 and batch ceiling 1 on SM80, with exact agreement against its independent
FP32 reference. The four prepared native configurations execute but retain
accuracy failures; see the family recipe for their separate profiles and results.
