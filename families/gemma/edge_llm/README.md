# Gemma4 Edge-LLM execution

Gemma owns complete-network offload to the official Edge-LLM 0.11.0 snapshot,
revision `95515c2f87fba8982db5a519f9022277667b3cc9`. The family maps the upstream
ONNX exporter, native component builders and persistent C++ runtime. No shared
model selection, cross-compilation or model-time installation is introduced.
Earlier Gemma generations keep their existing native implementation.

## Installation and build

Provision the [native SDK](../../../cmake/edge_llm/README.md) with
`TRTMC_EDGELLM_ALL_KERNELS=ON` and `TRTMC_EDGELLM_ONNX=ON`, configure Model
Connect with `TRTMC_ENABLE_EDGELLM=ON`, and expose the installation through
`CMAKE_PREFIX_PATH`. The 0.11 SDK prerequisite also installs the upstream
`tensorrt-edgellm[export]` Python dependencies. Pip tools do not replace the
C++ headers, archives, plugin and native builders required by this adapter.

```sh
# Standalone: export and build the checkpoint's text and available media towers.
trtmc gemma build /path/to/gemma-4-E2B-it --precision fp16 -o gemma.bundle

# Explicit text-only target/draft pair.
trtmc gemma build /path/to/gemma-4-12B-it --precision fp16 \
  --execution-variant mtp --companion draft=/path/to/gemma-4-12B-it-assistant \
  -o gemma-mtp.bundle
```

The family-owned `--execution-variant` values are `mtp`, `dspark`, `eagle3` and
`dflash`. The draft must be an existing local checkpoint with compatible
geometry; it is never inferred or downloaded. Options use the existing family
CLI protocol from #1310, not new shared flags or hooks. Python callers use
`GemmaBuildRequest`, `BuildExecutionInputs` and `NamedCheckpoint` from this
family's `edge_llm.config` with the unchanged public build API.

Standalone builds use FP16 compute, TP1 and batch1. Original checkpoints retain
their unquantized weights. The published 26B-A4B ModelOpt NVFP4 checkpoint also
keeps its packed weights on native SM12x; the dense 31B prepared route has
separate full-model accuracy results below; effective cache precision is read
from the built engine rather than inferred from the sidecar. No calibration or
quantization conversion is performed. They expose text
continuation and the image/audio tasks supported by the checkpoint's towers.
The runtime preserves ordered typed media and delegates preprocessing to Edge;
images are host RGB8 or normalized RGB float, and audio is mono 16 kHz float
PCM. Unsupported content, generation options and unmapped build overrides are
rejected rather than silently discarded. Paired builds are text-only and mapped
to native Linux x86_64 SM80. Other packed layouts and quantized companion pairs are not admitted by
this change. Each standalone build resolves only its actual media towers, using
the installed SDK's declared visual/audio executable paths.

The builder packages validated engines, tokenizer/chat-template files and
required ordinary/per-layer embeddings. It does not duplicate original weights
or ONNX intermediates into the bundle. Edge 0.11 processes the exported
`chat_template.jinja`; the old 0.10 static-template workaround is not carried
forward. Output comparison does not strip generated text or relax gates.

Ordinary Edge preparation failures retain a diagnostic log, warn and attempt
the family-owned native builder. Dense Gemma4 text, image, audio and combined-media input have native paths; modes
without an equivalent native graph remain explicit failures. A failed companion request never produces a base-only
bundle. Native Gemma4 rejects `fp32_layers`, `mtp_seq_len` and
`graph_transform` rather than silently dropping these unsupported controls.
Publication and runtime failures propagate without backend switching.
Use scratch-backed output and `TMPDIR` with enough space for export and bundle
extraction; these operations are bounded-memory, not zero-disk-space.

## Recorded 0.11 validation

These are prior local Model Connect build/inference results using native CUDA
13.3 / TensorRT 11.1.0.106, original weights, FP16, TP1/batch1 and the existing
family validation helpers. They are not new-PR-head CI or full-family approval.
No 0.10 result is used as evidence for 0.11. Successful and accuracy-failing
modalities are distinguished below.

| Checkpoint | Successful tested modes | Executable modes with a quality failure |
| --- | --- | --- |
| google/gemma-4-E2B-it | Standalone text and audio; MTP text with its official assistant | Standalone image |
| google/gemma-4-E4B-it | Standalone image | Standalone text/audio; MTP text with its official assistant |
| google/gemma-4-12B-it | Standalone text and image; MTP, DSpark, EAGLE3 and both DFlash drafts in greedy text mode | Standalone audio |

The 12B companion checkpoints exercised are:

- `google/gemma-4-12B-it-assistant` (MTP).
- `deepseek-ai/dspark_gemma4_12b_block7` (DSpark).
- `deepseek-ai/eagle3_gemma4_12b_ttt7` (EAGLE3).
- `deepseek-ai/dflash_gemma4_12b_block7` and
  `z-lab/gemma4-12B-it-DFlash` (DFlash block7 and block16).

Successful greedy companion cases matched the independent target-model HF
reference; speculative-activity checks distinguish actual draft execution from
fallback. Quality failures remain runnable and remain failures against the
unchanged owning criteria. This table does not qualify every prompt, media
shape, context length, platform, combined image/audio request or generation
policy. Original-weight 26B/31B variants and their assistants are not established by
these smaller-model results.

## Sampling and capacity

MTP remains restricted to greedy requests. DSpark preserves mapped sampling
controls. EAGLE3 and DFlash V1 sampled requests complete through Edge 0.11's
vanilla fallback, but fail the unchanged positive-speculative-activity gate.
Model Connect forwards those executable requests unchanged with a warning;
it does not claim speculative sampling passed or silently force greedy output.

The persistent runtime reserves decode lookahead and checks engine capacity.
For media, Edge performs preprocessing before it can count expanded input
positions, so the adapter conservatively reserves the whole input profile plus
lookahead. The upstream runtime rejects media beyond that profile. Unsupported
options or insufficient capacity are errors, not ignored requests.

## Published 26B-A4B NVFP4 standalone route

```sh
trtmc gemma build /path/to/Gemma-4-26B-A4B-NVFP4 --precision fp16 \
  --task images_text_to_text --max-sequence-length 2048 -o gemma-nvfp4.bundle
```

The published checkpoint revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` was built through Model Connect with
the original Edge 0.11 ONNX exporter and native SM120 text/visual builders.
Public Model Connect text and image commands completed. Direct Edge completed
the same-engine text and both original image fixtures, using 128 greedy tokens.
The unchanged upstream ROUGE-1/ROUGE-L gates (0.25/0.20) passed: text
0.3468/0.2197 and image 0.5825/0.3089.

**Quality remains unqualified:** Model Connect and direct Edge text and image outputs diverged
despite matching inspected greedy/thinking settings. The independent owning
HF FP32 NED comparison was not run because the installed reference loader does not
recognize this checkpoint's ModelOpt quantizer. Upstream ROUGE does not replace
that gate or resolve the divergence. These are executable requests, so the
adapter does not disable them for a quality failure. This evidence does not
qualify the 31B packed target, speculative companions, or other platforms.

## Native quantized attention precision

Earlier-generation Gemma native FP8 projections retain their calibrated Q/DQ
and compact weights. Attention score scaling, additive masking and softmax use
explicit FP32 TensorRT graph operations; probabilities return to the original
compute dtype before multiplication by V. This reuses the softcapped attention
math without applying a cap when the checkpoint does not specify one. Cache
and output dtypes are unchanged. Selection depends on quantization, not GPU
or TensorRT version; TensorRT still owns fusion and kernel selection.

## Original large-target native coverage

The original `google/gemma-4-31B-it` checkpoint at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` builds and runs through the
family-native FP16 image/text path on ARM SM103 / TensorRT 11.1, TP1/batch1,
with context capacity 2048. All six text comparisons (three cases, two repeats)
exactly match the original FP32 reference. Two image cases also match exactly;
the office-image response identifies the same object but differs in wording,
failing the unchanged NED 0.15 gate at approximately 0.3585. This remains a
runnable accuracy failure, not a builder or runtime failure.

The same original 31B target and its official MTP assistant at revision
`627c5ec1458b9086b841a91e0512fd31fd2fbbf1` also pass native FP16 text
qualification at context capacity 1024 on this platform. All six comparisons
exactly match both the original FP32 reference and ordinary decoding of the
same native target engine. Actual draft proposals, acceptance and verification
are checked; this is not base-only fallback. This does not qualify sampled
MTP requests, which remain explicitly unsupported.

The later one-row MTP verification correction has owning C++ regression
coverage for dense dual-profile and MoE single-profile targets, plus the full
26B replay reported below. The E2B/E4B/12B/31B full-model MTP results predate
that correction; they are retained evidence, not newly rerun qualification.

The family pair validator now accepts geometry-matched original dense and MoE
Gemma4 targets with their official MTP assistant or z-lab block16 DFlash draft.
The native draft graph reads both equivalent unscaled rotary-config spellings;
it does not reinterpret a Qwen-style companion as a Qwen target. Original
`google/gemma-4-26B-A4B-it` at revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` also passes native FP16 image/text
validation at context capacity 2048 on ARM SM103 / TensorRT 11.1. All three
image cases match the original FP32 reference exactly. Four short text
comparisons are exact; two long-text repeats have NED approximately 0.0481,
within the unchanged 0.15 gate. Its companion results are recorded below.

The original 31B target paired with
`z-lab/gemma-4-31B-it-DFlash` at revision
`eabd648301ce28583cc14757912e5e0f84e152e1` passes native FP16/context-1024
validation on ARM SM103 / TensorRT 11.1: six greedy comparisons pass the
unchanged original FP32 gate, and twelve seeded comparisons exactly match the
same native target and seeded sampler. Both modes exercise actual draft
acceptance and rejection. This is not cross-framework RNG parity.

The prepared dense `nvidia/Gemma-4-31B-IT-NVFP4` target at revision
`4135a98a9b728a548947683219633b25682223ac` has family-owned native TensorRT
FP4 weights, FP8 block scales and dynamic activation Q/DQ using its calibrated
scales. It does not substitute the routed experts' W4A16 behavior. A full native
FP16-compute/context-2048 image/text build and all nine C++ requests execute on
ARM SM103 / TensorRT 11.1. Against the retained original unquantized FP32
accuracy baseline, four short text comparisons and two image controls pass;
the office image has NED **0.3584905660377358**, and two long-text repeats have
NED **0.5363372093023255**, failing the unchanged **0.15** gate. The office-image
failure also occurs with the original native FP16 target. These remain runnable
accuracy failures, not passing quality claims or quantized-HF token parity.
The source has no serialized KV calibration scales; no FP8-cache qualification
is claimed. The independent small Q/DQ arithmetic checks remain separate from
these full-model accuracy results. No other platform is qualified by this run.

A separate 32-by-32 graph using this same dense W4A4 implementation reaches the
TensorRT 11.1 builder on an A30 (SM80), but TensorRT rejects its FP8 Q/DQ layers
because that GPU lacks the required hardware support. This is a demonstrated
native TensorRT precision constraint, not a family GPU blacklist or an OOM.
No engine or silent precision downgrade is produced. This probe does not
establish the result on SM89/L40s or qualify a full model on SM80.

## Native dense-text and media fallback

Gemma4 dense text now has a separate family-owned TensorRT Python graph and
uses the Gemma C++ runtime. It does not invoke an ONNX exporter or Edge runtime.
The graph preserves direct RMS scales, per-layer inputs, proportional rotary
embeddings, local/global attention geometry, sliding windows and shared K/V.
Only K/V-owning layers allocate persistent caches, with their actual widths;
earlier uniform-cache bundles remain readable.

The small independent FP32 graph checks cover prefill and decode past a sliding
window with 35 layers and 20 shared-K/V layers. They are mathematical regression
evidence, not full-checkpoint qualification. An initial FP16 build exposed a
TensorRT compiler SSA assertion with native IAttention. The same failure was
reproduced with small synthetic weights. Expressing the identical grouped
attention as ordinary TensorRT matmul/softmax operations avoids that compiler
path; TensorRT still owns all fusion and kernel selection. No GPU conditional
or numerical-gate change is used.

Original `google/gemma-4-E2B-it` revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` passed current-source native
FP16/context2048 validation on x86 RTX A6000 with TensorRT 11.1, including the
per-layer embedding representation described below. Three public CLI text
requests matched the retained original HF FP32 tokens exactly, including a
128-token continuation (NED 0 against the unchanged 0.15 owning gate).
Text replay of a combined-media bundle selects `--task text_continuation`.
Edge and BYOK were disabled. This uses the existing owning assertion, not a
newly registered full-family E2E.

Original `google/gemma-4-E4B-it` revision
`ee0ef6023621cff504d758262d4e04895a5af4a2` also passed all three native
FP16/context2048 checks on A30, with exact original FP32 reference tokens and
NED 0. The initial graph exposed a TensorRT large-Gather failure with the packed
per-layer token embedding table. An isolated original-weight Gather reproduced
an illegal memory access; the same rows in a per-layer table matched exactly.
The graph now represents each decoder layer's token embedding separately,
preserving weights and arithmetic without a GPU/version condition.

The native pipeline exposes the public text-continuation contract as well as
the existing text-generation interface. It maps explicit system messages and
the provider's Unicode trimming/thinking framing. Requests that exceed the
engine's actual position capacity are rejected before execution; unsupported
controls are never silently discarded. Both the current full-checkpoint E2B
result and the small original-reference checks cover the per-layer table
representation.

The native image path adds the checkpoint vision encoder and projector as a
separate TensorRT engine. CPU preprocessing preserves the provider's RGB8
bicubic resize, patch ordering and spatial positions; padding is excluded from
vision attention and pooling. Image features replace decoder input embeddings,
while per-layer token-identity embeddings still use the checkpoint's pad token.
The runtime preserves ordered images, including multiple images in one request,
through the existing public image/text Task. No GPU-specific helper kernel is
introduced.

The E4B native FP16/context2048 image bundle passed three public C++ API cases on
A30: a real scene question, a color control, and an ordered two-image question.
All tokens matched fresh original HF FP32 references (NED 0 against 0.15).
The same bundle also preserved the three dense-text cases, explicit system
message, raw prefix and capacity/unknown-option rejection checks. Independent
small original-reference tests cover both causal and local image-bidirectional
attention, including decode through shared K/V layers. These do not establish
image qualification for other Gemma4 checkpoints, precision modes or platforms.

The native audio path adds the checkpoint Conformer encoder and projector as
a separate TensorRT engine. CPU preprocessing preserves waveform framing,
periodic Hann windows, magnitude mel features and checkpoint normalization.
Audio must be finite mono 16 kHz PCM; incompatible rates/channels and capacity
overflow are rejected rather than resampled, relabeled or truncated. Encoder
validity excludes padding from soft-token expansion. Ordered image/audio parts
share the conditioned decoder; only image blocks can receive vision-only
bidirectional local attention. No GPU-specific audio helper is introduced.

The original E4B native FP16/context2048 combined bundle passed public C++ audio
transcription, language identification and combined image-color/audio-language
requests on A30, with exact original FP32 tokens and NED 0. Existing image and
text/continuation regressions also passed. CPU frontend checks matched original
features across 27 cases, including frame boundaries, preemphasis variants and
audio longer than 30 seconds. Small independent graph checks cover causal
audio conditioning even when vision-bidirectional attention is configured.

The current E2B FP16/context2048 path also passed seven public C++ media
requests on x86 RTX A6000 / TensorRT 11.1: three image cases, three audio cases,
and one combined image/audio case. Six matched original FP32 tokens exactly;
the ordered-audio comparison had NED 0.0408 against the unchanged 0.15 gate.
These are separate E2B checkpoint results, not inferred from E4B coverage.

**A semantic quality issue remains:** for two recordings, both original FP32
and native FP16 continued material from the first clip instead of transcribing
only the second short clip. Native/reference NED was 0.0809 against the unchanged
0.15 gate, but passing that comparison does not qualify instruction following.
The same concern occurs for E2B, despite its passing NED 0.0408 reference
comparison. The requests remain runnable; this is not a command-admission
restriction or evidence of an Edge-only defect.

The unified dense model has a separate family-owned media projection path:
merged RGB patches feed LayerNorm/dense/position projections, and raw 640-sample
waveform frames feed an audio projection. It does not reuse the E4B vision or
Conformer encoders. Serialized checkpoint aliases, merged-patch preprocessing,
and the unified non-thinking chat suffix are preserved.

The real unified image projection reaches about 86,810 before normalization,
above FP16's finite range. Both original PyTorch FP16 and naive native FP16
produce nonfinite features there. The native graph keeps that projection and
its immediately following LayerNorm in FP32 before returning to the requested
compute dtype. It neither clips activations nor imposes a GPU restriction.
Original FP32 and native FP32 component results agree; this does not establish
the cause of any earlier Edge-specific quality failure.

Original `google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` passed native
FP16/context2048 build and eight public C++ Task comparisons on an RTX A6000
with TensorRT 11.1. Text, image, audio transcription/language and combined-media
outputs all matched original full FP32 reference tokens exactly (NED 0 against
the unchanged 0.15 gate). Invalid audio rates, channels and frame overflow
were rejected. Five image/preprocessing/chat fixtures and eleven raw-waveform
fixtures also matched original processing exactly.

**Reference parity is not semantic success:** both original and native called
the second blue image Black, and both continued the first recording when asked
to transcribe only the second short recording. These remain recorded accuracy
issues and runnable requests, not successful instruction-following tests.

Routed-expert graph math passes small original-FP32 comparisons across
prefill, decode, shared caches and sliding windows. Prepared NVFP4 weights,
dense and ordered-vocabulary MTP, and DeepSeek block7 DFlash/DSpark have the
scoped full-checkpoint results below. EAGLE3 also has the scoped qualification
below. The z-lab block16 DFlash path also has scoped full-checkpoint and
independent component results below. Missing modes are not silently
converted to ordinary dense text. TP is outside the currently admitted Gemma
offload scope. This change does not claim complete Gemma4 fallback coverage,
other checkpoint qualification or cross-platform qualification.


### Native dense MTP fallback

The native verifier consumes each target token using the ordinary one-token
shape before committing its K/V state. The official assistant still proposes
and undergoes target acceptance/rejection; no native MTP speedup is claimed.
This avoids multi-row FP16 verification changing the accepted cache state and
therefore later target decisions. Greedy-only admission is unchanged.

For dense assistants, including ordered-vocabulary heads, an unavailable or failed Edge
preparation now falls back to a complete native target-plus-assistant bundle,
not a target-only substitution. The assistant reads the last owning target K/V
cache for each attention type and combines the pending target-token embedding
with the preceding verified target hidden state. The target verifies a genuine
assistant proposal causally and commits only the accepted prefix. The current
native runtime requires greedy sampling; stochastic verification is not mapped.

The original google/gemma-4-12B-it target and
google/gemma-4-12B-it-assistant revision
46d4c6f13f0ac0ad827b915669b8df9b81c64c51 built and ran natively with
FP16, TP1, batch 1, and context 1024 on an x86 RTX A6000 using TensorRT 11.1.
Five public text-continuation cases, each repeated on the same loaded model,
passed the unchanged original-FP32 NED 0.15 checks. The arithmetic, capital,
counting and translation cases matched the original tokens exactly. Raw-prefix
continuation had NED 0.125 and was not token-identical to FP32; the same native
target without speculation produced exactly the MTP tokens on all five cases.
The runs proposed 80 tokens and accepted 50, including real rejected proposals.
This is correctness evidence, not a speedup or all-assistant qualification.

Exact speculative equivalence is checked against the same native target;
the independent FP32 comparison retains the unchanged owning quality gate.
Existing family tests additionally cover acceptance, rejection, EOS, request
reset, and the final-token cache-capacity boundary. The public probe rejected
an actual overflow and unsupported stochastic verification.

The original google/gemma-4-E4B-it target and
google/gemma-4-E4B-it-assistant revision
8d0031ea8c2109e2b1e86bb9368a4539b537f80a also passed the native FP16,
TP1, batch 1, context 1024 path on x86 RTX A6000 / TensorRT 11.1. Its ordered
head selects vocabulary blocks by centroid, evaluates the selected embeddings
and restores canonical token IDs using TensorRT operations. The capital,
arithmetic and 128-token continuation cases each passed twice with exact
original FP32 tokens (NED 0) and exact same-native-target parity. These runs
proposed 174 tokens and accepted 88, exercising both acceptance and rejection.
No other assistant size, sampled decoding or platform is qualified by these
results.

The original E2B target with `google/gemma-4-E2B-it-assistant` revision
`2d874ef7d29f9a30599a1e4b3c1cbc9595f005df` builds and runs through the same
native FP16/TP1/batch1/context1024 path on x86 RTX A6000 / TensorRT 11.1.
All six repeated requests exactly match ordinary decoding of the same native
target; the runtime proposed 172 tokens and accepted 92, exercising acceptance
and rejection. **Independent FP32 quality is not fully qualified:** the four
capital/arithmetic comparisons have NED 0, but both 128-token continuation
repeats have NED 0.4807 against the unchanged 0.15 gate. This is a runnable
accuracy issue, not a reason to disable build/runtime or silently change
precision. Same-target equivalence does not replace the independent gate;
target numerics or build-profile sensitivity remain hypotheses, not a proven
Edge-only or speculative-verification defect.

### Packed-checkpoint precision audit

The adapter reports cache quantization from the built Edge engine's
`kv_cache_dtype`, not from ModelOpt's `hf_quant_config.json` alone. That sidecar
can declare FP8 without serialized LLM K/V calibration scales; Edge 0.11
detects those scales from checkpoint tensor names. An NVFP4 weight label does
not by itself establish FP8-cache execution.

The family-native routed graph maps the admitted 26B-A4B prepared checkpoint.
Its pinned Edge expert implementation uses NVFP4 weights with unquantized
activations; the source input-scale fields are not evidence of FP4 activation
execution. Native TensorRT operators decode one expert's compact source bits
inside an expert loop, preserving its block/global scales and source-order
accumulation. Calibration scales are constants, not runtime FP8 operators.

MoE bundles use one dynamic TensorRT profile for both prefill and decode.
The family reuses the existing backend module API, binds only cache inputs,
and appends actual module-owned output rows. This avoids two full context
allocations without changing shared core or introducing a GPU condition.
Existing lifecycle tests cover prompt/token shape transitions and request reset.

The published revision above built and executed through the public native
FP16/context256, TP1/batch1 path on x86 RTX A6000 with TensorRT 11.1. The engine
bundle was about 22.3 GB. This resolves the earlier native compiler and context
allocation failures for that exact slice; it is not a general memory guarantee.

**A native accuracy issue remains:** capital and arithmetic requests, each
repeated, matched an independent original-model FP32 reference exactly. Both
128-token continuation repeats instead had NED 0.1770 against the unchanged
0.15 gate, first diverging at generated token index 89. The original reference
uses the published packed weights independently decoded into the unmodified
Transformers model; it is not a comparison with an unquantized checkpoint.
The executable adapter remains available, but this is not a quality pass.
Other capacities, assistants and platforms are not qualified by these text results.


A controlled native FP32 diagnostic with the same source, context and platform
matched all six saved original-reference comparisons exactly (NED 0). This
supports precision sensitivity in FP16 on these cases; the exact lower-precision
boundary has not been isolated. The FP16 failure remains recorded. This does
not silently change requested precision or qualify another Edge configuration.

The same prepared source also passed the native FP16 image path at context 640,
TP1/batch1 on x86 RTX A6000 / TensorRT 11.1. The public CLI and the existing
real-scene, color-control and ordered two-image cases completed; all three
matched the independent original FP32 reference token-for-token (NED 0).
That reference includes the provider's image token-type IDs, required for this
target's bidirectional vision attention. No extra production-code change was
needed for this media slice. Its approximately 23.5 GB bundle contains the
native decoder and vision engine. This image result does not erase the
separate FP16 long-text accuracy issue or qualify other capacities/platforms.


### Native DeepSeek block7 drafts

The family-native block draft recomputes its context from committed target
features. It preserves the draft's own embedding, attention, MLP and language
head. DeepSeek DFlash proposes a bidirectional block; DSpark additionally
applies its sequential previous-token Markov correction and predicts confidence
from hidden state plus the Markov embedding. Both heads execute in TensorRT.
Confidence scheduling stays off, matching the paired Edge adapter's default;
threshold/SPS scheduling is not exposed by this native path.

The native verifier deliberately uses the ordinary target's one-token shape.
An initial multi-row verifier passed the independent FP32 quality checks but
failed exact same-native-target parity: a controlled same-engine comparison
found FP16 K/V and logits changed with the verification row count. Canonical
one-row verification fixes that parity failure without relaxing either gate.
Only verified tokens and target features enter persistent state. Actual block
proposals and accept/reject decisions remain; **no speculative speedup is
claimed**. The native block verifier now accepts mapped sampling controls,
using one target draw per emitted token. On rejection it preserves the already
sampled target token rather than advancing the RNG again. The draft still
executes deterministic proposals; no fast speculative-sampling claim is made.

With google/gemma-4-12B-it revision
707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 and
deepseek-ai/dflash_gemma4_12b_block7 revision
7490ce60c7630107917fe558e2bbe3dcec6195cb, the public FP16/context1024,
TP1/batch1 path passed on x86 RTX A6000 / TensorRT 11.1. Five text cases,
each repeated on the same loaded model, passed the unchanged original-FP32
NED 0.15 gate; raw-prefix continuation had NED 0.125 and the other cases
were exact. All ten outputs matched ordinary decoding of the same built
native target token-for-token. Acceptance, rejection, request reset, EOS,
capacity boundaries and seeded target sampling are covered by the existing
owning tests. These results do not qualify another draft architecture or GPU.


The same target with deepseek-ai/dspark_gemma4_12b_block7 revision
2fa72e765eec2965fc4d86a8663ce6769eba6218 also passed public native
FP16/context1024, TP1/batch1 on x86 RTX A6000 / TensorRT 11.1.
All ten outputs passed the unchanged original-FP32 gate (worst NED 0.125)
and matched the same built native target exactly. The runs made 364 actual
proposals, accepted 84 and executed both the Markov and confidence heads.
Four independent original-FP32 component cases also matched proposal IDs
exactly and passed unchanged numerical tolerances for corrected logits,
confidence and hidden states, including short blocks and large fused features.
Confidence scheduling remains off. These original greedy results do not
qualify sampling; the separate sampled validation is described below. No
speculative speedup is claimed.


### Native Gemma EAGLE3 fallback

The single-layer EAGLE3 graph combines each committed target feature with the
next token's scaled embedding, retaining the feature's original position.
It recomputes committed draft K/V and unrolls seven recurrent proposals using
its own language head. Rejected states never enter persistent target history.

The engine exports a fixed seven-row output bound; the runtime consumes only
the prefix that fits target capacity. This fixes an initial native integration
failure where decreasing output size at the maximum input profile caused
insufficient output allocation. The correction is family-owned, not a shared
backend or upstream TensorRT change.

With the same 12B target and deepseek-ai/eagle3_gemma4_12b_ttt7 revision
0bc24c312350910419cf371e54082f040d65cc82, public FP16/context1024,
TP1/batch1 build passed on x86 RTX A6000 / TensorRT 11.1. Following the
output-contract fix, only the companion was rebuilt from the production family
graph; the original target engine was retained and its copied SHA256 verified.
Public inference and all ten text comparisons then passed the unchanged
original-FP32 NED 0.15 gate (worst 0.125) and exact same-native-target parity.
The runs proposed 322 tokens and accepted 90. Five independent original-FP32
component cases also passed, including short valid prefixes and large fused
features. These earlier checks cover greedy decoding; sampled validation is
reported below. No speculative speedup is claimed.


### Native z-lab block16 DFlash fallback

The z-lab/gemma4-12B-it-DFlash revision
541d351de6dc9cd698fc348612bd5cdf03c636d0 uses its own five-layer
draft topology, unscaled target embeddings and tied target language head.
The first four layers preserve causal sliding attention and the final layer
uses full bidirectional attention. Block16 includes the pending anchor; the
runtime discards the anchor output and verifies at most fifteen proposals.
Its checkpoint architecture does not register a separate Qwen target adapter.

With the same 12B target, FP16/context1024, TP1/batch1, public build and
inference passed on x86 RTX A6000 / TensorRT 11.1. All ten greedy comparisons
passed unchanged original-FP32 NED 0.15 (worst 0.125) and exact same-native-target
parity. The runs proposed 750 tokens and accepted 84. Four independent
original-FP32 component cases also passed, including short blocks, sliding
windows and large features.

A fresh public paired build with the native sampling correction passes the same
ten greedy quality checks and twenty seeded comparisons against ordinary
decoding of its exact native target engine. Controls cover temperature 0.8,
top-k 50, top-p 0.95, seed 7; and temperature 1.0, unrestricted top-k,
top-p 0.9, seed 19, each repeated on the loaded model. Raw-prefix outputs differ
from greedy and remain token-identical to the seeded target control. The runs
proposed 1,590 tokens and accepted 162. This verifies native target sampling,
not token equality with another framework's stochastic sampler. Owning C++
sampling regressions cover all four implemented block variants. No other
platform or speculative speedup is established.

### Native sampled pairs: 12B full-checkpoint results

Fresh public paired builds for the same 12B target with DSpark block7, EAGLE3
TTT7 and DeepSeek-DFlash block7 also passed native FP16/context1024,
TP1/batch1 on x86 RTX A6000 / TensorRT 11.1. Each pair passed all twenty
seeded comparisons against ordinary decoding of its exact native target,
using the two parameter sets above and two repeats of five requests. The
public native CLI and C++ Task executed, including capacity rejection.

| Draft | Exact sampled comparisons | Proposals | Accepted | Verification rounds |
| --- | ---: | ---: | ---: | ---: |
| DSpark block7 | 20/20 | 686 | 172 | 98 |
| EAGLE3 TTT7 | 20/20 | 644 | 178 | 92 |
| DeepSeek-DFlash block7 | 20/20 | 630 | 180 | 90 |

All three exercised genuine draft proposals, acceptance and rejection.
Raw-prefix outputs differ across seeds and from greedy output, but match the
same-native-target sampled controls exactly. Previously passed independent
FP32 greedy quality checks remain unchanged; these sampled comparisons do
not claim equality with another framework's random-number sequence. No
production-code or gate change was needed for these additional runs.

This closes sampled validation only for the four listed 12B pairs. E2B now
has its own standalone and runnable MTP-accuracy-failure evidence above.
The larger original-checkpoint results are recorded separately above and below;
these 12B comparisons do not qualify another target, capacity or platform.

### Native original 26B-A4B companions

The original `google/gemma-4-26B-A4B-it` target at revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` and its official assistant at
`6e5aaaf4c42b98394530b8fda2e95cadd65c151c` passed a native FP16 paired build
and six C++ requests with context 1024, TP1 and batch1 on ARM SM103 / TensorRT
11.1. All six outputs exactly matched both the original FP32 reference and
ordinary decoding of the same native target. Real assistant acceptance and
rejection were exercised. The one-row verifier described above corrected a
batched accepted-cache drift; the accuracy threshold was not changed.

The same target with `z-lab/gemma-4-26B-A4B-it-DFlash` at revision
`77d4202772dfe50b2396ec7bac9cfffc7b9e7057` also builds and executes natively.
All 18 requests exactly matched ordinary decoding of the same serialized target
and native sampler: six greedy and twelve seeded requests. Four short greedy
comparisons exactly matched original FP32; two long repeats had NED
**0.2854838709677419**, failing the unchanged **0.15** gate. This is a runnable
native-target accuracy issue, not a demonstrated draft-verification or Edge
command failure. It is not claimed as an accuracy pass or a speedup; no other
platform or cross-framework sampling equivalence is established.

### Native published QAT-original 26B-A4B

`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` at revision
`f1e06dc520982d9b9edd76859fdb7ab209449949` passes a native FP16 image/text build
and all nine C++ requests at context 2048, TP1 and batch1 on ARM SM103 /
TensorRT 11.1. Six text comparisons and three image comparisons exactly match
an independent FP32 eager reference using this same published checkpoint;
TF32 is disabled and the unchanged NED gate is 0.15. This exercises the
published unquantized QAT weights, not newly calibrated or packed Q4_0 inference.
These are scoped native results, not a new Edge-engine or other-platform claim.
