# Qwen3.5/3.6 MoE: Edge-LLM execution

This family owns checkpoint admission, builder selection, bundles, runtime
orchestration and validation. It does not add model-specific behavior to shared
Model Connect code. On an Edge route, Edge owns the complete network, quantized kernels and
speculative algorithm. Native fallback graphs and orchestration stay in this family.

## Installation

For the default Edge SDK, **0.12.0**, build on
the executing target with `-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.12.0`.
For all native kernel groups, add `-DTRTMC_EDGELLM_ALL_KERNELS=ON` to a pinned
source build. The official 0.12 revision is `513dcd18291e587c80928911a480761352f3c2d8`.
Expose the installed SDK through `CMAKE_PREFIX_PATH`; do not mix the C++ ABIs
or cross-compile. See the [CMake provisioning guide](../../../cmake/edge_llm/README.md).

The Python builder may come from the matching PyPI package or a matching source
installation. To use an isolated exporter, pass
`--edge-builder-python /path/to/edge-env/bin/python` to the family build.
The adapter checks the builder version and local TensorRT version before use.
Installing the Python wheel alone does not supply Model Connect's compiled C++
family runtime. Model builds do not install packages or upgrade dependencies.

`edge_llm/builder.py` maps the owning request into the upstream Python builder.
`runtime/edge_llm/` loads the complete result through the upstream C++ runtime.
The adapter verifies native GPU, architecture, CUDA and TensorRT compatibility.

## Checkpoint and precision contracts

| Checkpoint | Preserved source recipe |
| --- | --- |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | Group128 GPTQ routed experts; original shared experts and attention; FP16 KV and vision |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | Group16 NVFP4 W4A16 routed/shared experts and LM head; FP8 attention/GDN projections; FP16 KV and vision |

In 0.12, the adapter also maps original BF16-weight Qwen3.5 35B/122B and
Qwen3.6 35B checkpoints to explicit FP16 compute. The prepared Qwen3.5 122B
NVFP4 source has a different layout from the 35B mixed source: its routed
experts use calibrated W4A4; its shared expert, LM head and attention retain
original weights. It is not the 35B W4A16 recipe scaled up.

The published original-weight checkpoints store fused expert tensors.
The 0.12 direct builder expects per-expert tensors. Enable
`-DTRTMC_EDGELLM_ONNX=ON` to use the released ONNX exporter and the installed
SDK's native compilers on the executing GPU. The bundle contains the compiled
engines and their exported weight files, not a second copy of the source
checkpoint. Packed sources retain the direct builder. For a documented original
Qwen3.6 35B DFlash or JetSpec pair, the adapter supplies the matching draft to
the base and draft exporters and compiles both roles into the same engine
directory. A successful draft export alone does not qualify a speculative pair.

These routes use FP16 compute, one batch and one device. The NVFP4 Edge route requires a
compatible Blackwell target. Native TensorRT owns hardware lowering; unavailable
quantized operations are errors, never an automatic precision substitution.
`--quantization`, when supplied, asserts the source format; it does not calibrate,
rewrite or silently substitute another checkpoint. Unknown or conflicting
precision metadata is rejected before compilation.

The published NVFP4 checkpoint labels KV quantization as FP8 in its sidecar but
has no exported language-model K/V scales. The pinned Edge resolver therefore
uses FP16 KV, as does native fallback. A sidecar label alone is not evidence of
FP8 language-model cache storage. Prepared sources with actual exported scales
retain FP8 KV rather than silently widening storage.

Accepting source metadata or executing a single routed layer does not qualify a
full checkpoint. Build, inference and independent quality must be assessed per
checkpoint, modality and draft; memory on multiple GPUs is not available to a
single-device engine.
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
visual engines. The family preserves the provider template. Edge routes delegate image
preprocessing to Edge; native routes use the owning media implementation. The profile permits 4–512 tokens per image and at most
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

## Paired drafts

The family CLI owns `--execution-variant` and `--companion`. Supply a local,
explicitly matched draft; no companion is inferred or downloaded automatically.

| Target | Companion |
| --- | --- |
| Qwen3.5 GPTQ 35B | `z-lab/Qwen3.5-35B-A3B-DFlash` |
| Qwen3.6 NVFP4 35B (or, in 0.12, the original 35B) | `z-lab/Qwen3.6-35B-A3B-DFlash` |

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

### JetSpec (0.12)

The documented `JetSpec/jetspec-Qwen3.6-35B-A3B` companion pairs with the
original Qwen3.6 35B or its mixed NVFP4 checkpoint. Use
`--execution-variant jetspec --companion draft=/models/jetspec-Qwen3.6-35B-A3B`.
This is a different trained draft, not a DFlash option: each proposal row may
attend to confirmed target features and earlier or current proposal rows, but
not future rows. Both backends use greedy top-7 branching and 16 verification
nodes including the root. A 0.11 SDK cannot offload JetSpec. These mappings
describe the execution contract, not independent accuracy results.

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
`image_video_text_to_text` through the existing Task contracts. Edge routes pass
ordered RGB frames to Edge's video frame stack; native routes own preprocessing,
temporal patch grouping and token expansion. Both preserve explicit timestamps.
Absent timestamps use the official frame-list default of one FPS. Frames must
have matching dimensions; invalid timestamps and storage are rejected. No shared
parser or new video CLI is added.

Small frame-based transport and rejection/recovery checks are not evidence of
motion understanding or independent video accuracy. Speculative media
qualification is separate from text-only and ordinary-image results.

## Failure and quality policy

When no Edge platform route applies, this family constructs the same request
with its native TensorRT Python graph and Model Connect C++ runtime. Edge
preparation errors retain diagnostics, warn, and fall back once before bundle
publication. The fallback preserves the source quantization, media capacity,
and requested embedded MTP or explicit DFlash/DDTree companion; it does not
silently substitute ordinary decoding. Publication errors and cancellation do
not trigger fallback. Runtime errors are returned without retrying through
another implementation.

The native routed-expert graph gathers the selected experts before decoding their
original packed weights, then accumulates in ascending expert-index order. It
preserves source quantization without evaluating every unselected expert. Target
and embedded-MTP engine construction permits up to eight CPU builder threads,
capped by the reported CPU count; this does not select GPU tactics or change precision.

Native full-checkpoint and speculative qualification is tracked separately from
Edge command results. Small operator and integration checks alone do not qualify
the documented 35B checkpoints.

Full native MTP is not yet qualified: one full 35B build exceeded a 90-minute
validation limit while compiling its four-position target verifier, after the
single-token target and embedded MTP engines had compiled. No full bundle or
inference result was produced. This is a native build-time limitation, not an
Edge builder failure or a proven TensorRT defect. The requested three draft
steps and four verification positions remain unchanged; they are not replaced
with ordinary decoding.

**Measured 0.12 results (single device, FP16 compute, 1,024-token context).**
With `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, both Edge and native TensorRT
matched the independent packed Hugging Face reference on the tested ordinary
raw-text and image prompts. With its explicit DFlash draft, both runtimes
matched the raw-text reference and the released Edge runtime's chat response;
this is not a paired-image result.

For the original `Qwen/Qwen3.5-35B-A3B`, the MC Edge ONNX and native
vision bundles both matched the independent BF16 Hugging Face token IDs on
raw text, chat and image. The released Edge runner returned the same text
on all three requests with the MC-built engines.

For the original `Qwen/Qwen3.6-35B-A3B`, use the ONNX flow. The ordinary
vision bundle matched the independent BF16 Hugging Face token IDs on all three
32-token requests: raw text, chat and image. On native TensorRT, the DFlash
and JetSpec pairs each matched the ordinary native token IDs on raw text and
chat. That verifies the greedy target comparison, not reference accuracy: both
native pairs scored 0.644 on raw text and 0 on chat (limit 0.15).
The original DFlash pair also built and ran through the Edge ONNX route.
Its chat matched the reference exactly; its raw completion scored 0.644 and
differed from ordinary Edge decoding. Stock Edge reproduced the paired output
on the same engines, and native DFlash returned the same raw token IDs. The
pair is executable, but the raw result is not accuracy-qualified.
On the original checkpoint, Edge ONNX with the published JetSpec draft
matched the independent and ordinary greedy token IDs on both raw and chat
requests. Stock Edge matched both outputs and its profile recorded actual
draft proposals. This JetSpec result is for text only; it does not qualify
native raw decoding or the mixed NVFP4 target.

With `nvidia/Qwen3.6-35B-A3B-NVFP4` and the documented JetSpec draft, both
runtimes built and ran. Both matched the independent mixed-precision reference
on a chat request. On an untemplated completion, normalized edit distance was
0.725 for Edge and 0.644 for native TensorRT (limit 0.15). The released Edge CLI
reproduced the adapter's response on the same engines. This rules out the MC
request mapping as the sole explanation for that difference; it does not
establish which runtime is at fault. These routes remain executable.

The same NVFP4 target with its documented DFlash draft also built and ran on
both routes. Both matched the chat reference exactly, but the untemplated
completion exceeded the same gate: 0.521 on Edge and 0.644 on native TensorRT.
These are text-only comparisons, not evidence of paired media accuracy.

Without a draft, the installed Model Connect wheel built both text and visual
Edge bundles. Edge and native TensorRT both ran raw text, chat, and an image.
Both matched the independent chat reference; neither met the same 0.15 gate on
the raw completion (0.725 Edge, 0.644 native) or image description (0.704 Edge,
0.531 native). The standalone Edge CLI returned three distinct completions for
three identical raw requests on the same engine with batch size 1, temperature
0, top-k 1, and top-p 1. Their edit distances were 0.685, 0.058, and 0.725.
A single passing standalone run is therefore not a reproducible accuracy pass.
This control does not identify a TensorRT defect or explain the native result.

On the installed mixed NVFP4 vision bundle, the public CLI also accepted
requests without an image. In that run the raw, chat and image scores were
0.064, 0 and 0.719, respectively. The released CLI on the identical engines
also missed the image gate (0.682). Neither this run nor the earlier text-only
run establishes repeatability.

When comparing a vision bundle with the released `llm_inference` CLI, pass
`--multimodal-engine-dir` for **all** requests, even those without images.
Model Connect loads the visual runtime when it opens the bundle. On the
prepared Qwen3.5 122B NVFP4 model, the stock raw completion differed when
the visual engine was omitted; loading it reproduced the MC completion.
For independent validation, vLLM used the same prepared weights with CUTLASS
experts, FP16 compute and FP16 KV. The MC Edge bundle matched its raw and
chat token IDs exactly. The image result missed the 0.15 gate (NED 0.475);
the released Edge runner produced the same image response on the same engines.
Native TensorRT also matched the raw and chat reference IDs, but its image
result missed the gate (NED 0.539). Both image routes remain executable.
The original, unquantized Qwen3.5 122B checkpoint was not built or run:
no validation destination had room for the estimated staged build outputs.
That is a storage limitation of the experiment, not an inference OOM. Results
for the prepared NVFP4 checkpoint do not qualify the original weights.

**Command success is not accuracy qualification.** The recorded DFlash outputs
can differ from same-checkpoint ordinary Edge decoding beyond the existing
0.15 normalized-edit-distance gate. NVFP4 also has recorded public/direct and
repeated-image disagreement; successful commands do not resolve that numerical
issue. MTP's recorded ordinary-decoding comparison is an accuracy failure too.
Accuracy-only failures remain executable;
they must not be reported as passing quality validation.

Earlier independent packed-GPTQ CPU reference runs forced FP16 despite the
checkpoint's declared BF16 arithmetic. Raw generation degenerated despite exact
checkpoint-tensor verification, and image generation timed out. The reference
now follows the declared precision, including floating image inputs and precision
evidence. The official backend, source verification, requests and 0.15 gate are
unchanged. Correcting this setup mismatch alone does not establish the cause of
the earlier failure or qualify image accuracy; that still requires a completed
independent comparison, not only agreement between Edge and native execution.

The pinned symmetric GPTQ checkpoint omits its format field but stores literal
zero points of eight (GPTQ v2). The default official loader assumed v1 and added
an unintended offset. The reference now checks every packed zero-point tensor,
rejects mixed formats or conflicting declarations, and supplies the inferred
format through the official loader API without modifying checkpoint files.
Source verification applies the v1-to-v2 offset only for actual v1 sources.
This preserves the original source-verification and independent quality gates;
format controls alone do not establish full-model accuracy.

The selected GPU E2E runs the same independent GPTQModel Torch reference on the
leased CUDA device, after the Model Connect subprocess has exited. It does not
hide the GPU or choose placement based on the comparison result. The CPU
reference produced a substantially different image description in a recorded
comparison, while a source-verified CUDA control matched both frozen 32-token
outputs exactly in the same request order. This isolates a reference-environment
discrepancy, not its underlying CPU kernel or version cause. Explicit
`--device cpu` remains available for diagnosis. Both placements retain checkpoint
precision, source-tensor verification, offline inputs and the same 0.15 gate;
CUDA unavailability is an error, not a skip or an implicit CPU fallback.
A passing retained-output control is not a fresh Model Connect E2E result.

The owning CTest factory target also builds the standalone E2E drivers. These
helpers remain in the configured native build tree, not in an installed runtime.
Selected E2E tests use `TRTMC_NATIVE_BUILD_DIR` for helpers while retaining
`TRTMC_RUNTIME_ROOT` for loading the isolated runtime. Standalone
`families.qwen3_5_moe.tests.edge_validation` accepts `--native-build-dir` for the
same split layout; omitting it retains the colocated build/runtime layout.

Owning tests retain exact public/direct token comparisons when Edge is selected.
Native bundles instead exercise the owning public runtime and actual media task
bindings; they do not claim direct Edge parity. Both paths retain control
rejection, capacity enforcement, repeatability after errors and the independent
packed-HF quality gate. Full MMLU is additional evidence, not a replacement for that gate.
Qualification is scoped to the actual checkpoint, profile and workload; neither
unit tests nor representative results establish catalog-wide support.
