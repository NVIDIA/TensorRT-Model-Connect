# Qwen3.5/3.6 MoE: Edge-LLM 0.11 execution

This family owns checkpoint admission, builder selection, bundles, runtime
orchestration and validation. It does not add model-specific behavior to shared
Model Connect code. Edge owns the complete network, quantized kernels and
speculative algorithm.

## Installation

Native CMake provisioning uses `TRTMC_ENABLE_EDGELLM=ON` and the official public
Edge-LLM **0.11.0** revision `95515c2f87fba8982db5a519f9022277667b3cc9`.
Expose the installation through `CMAKE_PREFIX_PATH`. Build on the executing
target; there is no cross-compilation.

Compatible installed Python tools, including `tensorrt-edgellm==0.11.0` from
PyPI, are discovered automatically. Otherwise the adapter uses the native SDK's
source-installed Python. The wheel does not replace the native C++ headers,
archives and plugins. Model builds never install packages or select another
Edge release automatically.

`edge_llm/builder.py` maps the owning request into the upstream Python builder.
`runtime/edge_llm/` loads the complete result through the upstream C++ runtime.
The adapter verifies native GPU, architecture, CUDA and TensorRT compatibility.

## Checkpoint and precision contracts

| Checkpoint | Preserved source recipe |
| --- | --- |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | Group128 GPTQ routed experts; original shared experts and attention; FP16 KV and vision |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | Group16 NVFP4 routed/shared experts and LM head; FP8 attention/GDN projections and KV; FP16 vision |

Both use FP16 compute, one batch and one device. NVFP4 requires a compatible
Blackwell target. `--quantization`, when supplied, asserts the source format;
it does not quantize, dequantize or silently substitute another checkpoint.
Unknown or conflicting precision metadata is rejected before compilation.

The full original-weight 35B and 122B checkpoints and the 122B NVFP4 checkpoint
are outside the current validated single-device admission. Aggregate memory
across multiple GPUs does not establish a supported MoE parallel route.
Standalone Qwen3 and older checkpoints do not belong to this family.

## Build and run

```bash
trtmc qwen3_5_moe build /models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --precision fp16 --max-sequence-length 4096 -o qwen35-moe.bundle
trtmc run qwen35-moe.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

Use `--task vision_language_generation` for a bundle containing both the LLM and
visual engines. The family preserves the provider template and delegates image
preprocessing to Edge. The profile permits 4–512 tokens per image and at most
1024 image tokens per request. Expanded media tokens plus the requested output
budget must fit the bundle's context; early EOS does not excuse over-capacity
requests.

```bash
trtmc qwen3_5_moe build /models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --task vision_language_generation --precision fp16 \
  --max-sequence-length 4096 -o qwen35-moe-image.bundle
trtmc run qwen35-moe-image.bundle --image /images/dog.jpg \
  --prompt "Describe this image." --max-new-tokens 32 \
  --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

Raw text mode preserves prompt bytes, including trailing newlines. Image Task
API requests require the provider chat template. Unsupported generation
controls are rejected, not silently ignored.

## DFlash companions

The family CLI owns `--execution-variant` and `--companion`. Supply a local,
explicitly matched draft; no companion is inferred or downloaded automatically.

| Target | Companion |
| --- | --- |
| Qwen3.5 GPTQ 35B | `z-lab/Qwen3.5-35B-A3B-DFlash` |
| Qwen3.6 NVFP4 35B | `z-lab/Qwen3.6-35B-A3B-DFlash` |

These listed pairs use original BF16 drafts, including the NVFP4 target. The
adapter checks target width, vocabulary, layer selection and the block16
contract rather than imposing a same-precision requirement on target and draft.

```bash
trtmc qwen3_5_moe build /models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --execution-variant dflash \
  --companion draft=/models/Qwen3.5-35B-A3B-DFlash \
  --precision fp16 --max-sequence-length 1024 -o qwen35-moe-dflash.bundle
```

`dflash_ddtree` selects greedy top-4 tree drafting with 16 verification positions.
The pinned hybrid DFlash base contains the recurrent replay metadata required
by both linear and tree verification. Non-greedy DDTree requests are rejected.
### Embedded MTP

The documented Qwen3.6 NVFP4 35B MTP recipe uses the prediction layer embedded
in the target checkpoint, not a separate companion. The family maps it to
`--spec-type mtp`, three drafting steps and four verification positions.
GPTQ MTP is not admitted by this recipe.

```bash
trtmc qwen3_5_moe build /models/Qwen3.6-35B-A3B-NVFP4 \
  --task vision_language_generation --execution-variant mtp \
  --precision fp16 --max-sequence-length 4096 -o qwen36-mtp.bundle
```

### Typed video input

Visual bundles expose `images_text_to_text`, `video_text_to_text` and
`image_video_text_to_text` through the existing Task contracts. The family maps
ordered RGB frames to Edge's video frame stack, preserving explicit timestamps.
Absent timestamps use the official frame-list default of one FPS. Frames must
have matching dimensions; invalid timestamps and storage are rejected. Edge
owns temporal preprocessing and token expansion. No shared parser or new video
CLI is added.

The small repeated-frame smoke input checks video transport only. It is not
evidence of motion understanding or independent video accuracy. Speculative
media qualification is separate from text-only and ordinary-image results.

## Failure and quality policy

Compilation errors abort atomic bundle publication and retain diagnostics.
This owner has no native graph fallback: it warns and reports that fact rather
than silently changing the requested model or execution variant. Runtime errors
are returned without retrying through another implementation.

**Command success is not accuracy qualification.** The recorded DFlash outputs
can differ from same-checkpoint ordinary Edge decoding beyond the existing
0.15 normalized-edit-distance gate. NVFP4 also has recorded public/direct and
repeated-image disagreement; successful commands do not resolve that numerical
issue. MTP's recorded ordinary-decoding comparison is an accuracy failure too.
Accuracy-only failures remain executable;
they must not be reported as passing quality validation.

The independent packed-GPTQ CPU reference is not yet qualified: its recorded
raw generation degenerated despite exact checkpoint-tensor verification, and
image generation timed out. This leaves independent image accuracy unproven;
it is not evidence that the failure belongs exclusively to Edge. The existing
reference and 0.15 gate remain unchanged while that reference issue is diagnosed.

Owning tests retain exact public/direct token comparisons, control rejection,
capacity enforcement, repeatability after errors and the independent packed-HF
quality gate. Full MMLU is additional evidence, not a replacement for that gate.
Qualification is scoped to the actual checkpoint, profile and workload; neither
unit tests nor representative results establish catalog-wide support.
