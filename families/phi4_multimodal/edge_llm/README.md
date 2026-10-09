# Phi4 Multimodal with Edge-LLM 0.11

This family owns Edge selection, builder arguments, engine assets, runtime
orchestration, and validation. Shared model-selection code is unchanged.
The integration targets the official public Edge revision
`95515c2f87fba8982db5a519f9022277667b3cc9`.

## Installation

Provision the optional native Edge SDK through Model Connect's CMake dependency
and expose its install prefix through `CMAKE_PREFIX_PATH`. A compatible
`tensorrt-edgellm==0.11.0` Python wheel in the active interpreter is preferred
for Python builder/export commands. Otherwise, the source-provisioned SDK
interpreter is used. The wheel does not replace the native SDK.

Build engines natively on the target GPU; the adapter does not cross-compile.
The SDK and runtime must match the bundle's CUDA, TensorRT and GPU architecture.

## Documented model and routes

The checkpoint is `microsoft/Phi-4-multimodal-instruct`. Edge's documented
Phi4 recipe merges the vision LoRA before export. The original-checkpoint
experimental builder applies that vision adapter; the ONNX route consumes an
already merged and quantized checkpoint. Neither route claims to preserve the
separate unmerged Hugging Face LANGUAGE-mode weights.

| Checkpoint weights | Edge route | Native target | Build request |
| --- | --- | --- | --- |
| Original FP16 with vision LoRA | Experimental builder | SM80 or SM120 | FP16, TP1, batch1 |
| ModelOpt FP8 | ONNX | SM120 | FP16 execution, TP1, batch2, context8192 |
| ModelOpt NVFP4, with or without NVFP4 LM head | ONNX | SM120 | FP16 execution, TP1, batch2, context8192 |
| ModelOpt W4A16-AWQ | ONNX, ALL_KERNELS SDK | SM80 | FP16 execution, TP1, batch2, context8192 |

Use the existing `--precision fp16`, `--max-sequence-length`,
`--max-batch-size`, and `--tensor-parallel-size` build arguments.
Packed checkpoints must retain matching ModelOpt metadata; the adapter neither
calibrates nor changes packed weights. The quantized LM-head route avoids
unsupported externalized LM-head export.

Original-checkpoint image capacity is bounded by context, with a maximum of
1280 tokens per image and a 256-token reserve for separators and prompt tokens.
Packed routes use input7168/KV8192 and visual256..6400/per-image1280 profiles.
The runtime also validates actual expanded input plus the original generation
budget; arbitrary prompts and images are not guaranteed to fit.

## Runtime and quality boundaries

The family implements image-to-text and no-image text requests through the
same vision-adapted engine. Image input uses checkpoint chat framing; no-image
input preserves the family's raw-prompt behavior. Both checkpoint stop tokens
199999 and 200020 are retained. Audio is listed in Edge's model catalog, but
the pinned builder lacks a Phi4 audio component and this Model Connect family
has no audio task adapter. Audio is not exposed by this integration.

The existing image validation retains its fixture, prompt, 30-token budget,
FP32 reference, actual-feature health check, and NED limit of 0.15. Recorded
FP16 context8192 text output meets that limit. The context768 and context2048
runs and the four packed-weight configurations execute but fail that accuracy
gate. They remain runnable for investigation; accuracy outcomes never control
dispatch. A larger-context pass does not qualify the unchanged context768 test.

If Edge preparation fails, diagnostics are retained and the family warns before
retrying its native builder with the unchanged request. An incomplete Edge
bundle is never published. Runtime inference errors are returned to the caller
and never silently retried.
