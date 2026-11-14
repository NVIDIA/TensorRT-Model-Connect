# Qwen3.6 / Qwen3.8 Edge-LLM execution

This family owns checkpoint admission, build options, command mapping, bundle
assets, runtime orchestration, and validation. Shared code contains no Edge
selection. The optional SDK uses official Edge-LLM **0.11.0**, revision
`95515c2f87fba8982db5a519f9022277667b3cc9`.

A compatible installed Python wheel is preferred. Discovery checks the actual
builder or ONNX exporter needed by the request, rather than accepting a version
string alone. Matching source-installed SDK Python is the fallback. Python
package discovery does not replace the native CMake SDK dependency. Builds are
native; cross compilation is not supported.

## Executed Edge configurations

| Source checkpoint | Execution | Local build profile |
| --- | --- | --- |
| `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B` | Ordinary | Original weights, FP16, TP2, batch 1, Linux x86_64 SM80 |
| `RadixArk/Qwen3.8-27B-NVFP4` | Ordinary | Original mixed FP8/NVFP4 weights, FP16 execution, TP1, batch 1, Linux x86_64 SM120 |
| Same mixed checkpoint + `RadixArk/Qwen3.8-27B-DSpark` | DSpark | Unquantized block-7 draft, text, FP16, TP1, batch 1, SM120 |
| Same mixed checkpoint + `z-lab/Qwen3.8-27B-DFlash2` | DFlash2 | Unquantized block-8 draft, text, FP16, TP1, batch 1, SM120 |

Ordinary media builds expose existing image, ordered-image, video, and mixed
media Task interfaces. Availability of a binding does not qualify every media
workload. Current executed evidence and remaining gaps are listed below.

### Additional upstream execution mappings

The adapter also maps the following Edge 0.11 configurations. These have owning
Python and native integration coverage, but **no full prepared-checkpoint Edge
engine execution or accuracy result**:

- Qwen3.6 uniform NVFP4 with embedded MTP.
- Qwen3.6 INT4-AWQ with embedded MTP.
- Qwen3.8 INT4-AWQ with the documented DFlash2 companion.

These mappings require compatible ModelOpt checkpoint metadata: uniform NVFP4
with group size 16, or INT4-AWQ with group size 128. They do not accept every
checkpoint whose name contains NVFP4 or INT4. The adapter forwards prepared
weights to Edge; it does not perform calibration or create quantized checkpoints.
The locally executed table above is evidence scope, not a checkpoint-ID allowlist.

## Build and run

Provision the [native SDK](../../../cmake/edge_llm/README.md), configure its
installation in `CMAKE_PREFIX_PATH`, and compile with `TRTMC_ENABLE_EDGELLM=ON`.
The family options use the existing `cli.json` protocol, not a shared parser
extension. Help works offline:

```sh
trtmc qwen3_8 build --help

CUDA_VISIBLE_DEVICES=0,1 trtmc qwen3_8 build /models/Qwen3.6-27B \
  --precision fp16 --tensor-parallel-size 2 --max-sequence-length 4096 \
  -o qwen36.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 \
  --task vision_language_generation --precision fp16 --quantization nvfp4 \
  --max-sequence-length 4096 -o qwen38-media.bundle

trtmc run qwen38-media.bundle --image /path/to/image.png \
  --prompt "Please describe the image." --max-new-tokens 128 \
  --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 --precision fp16 \
  --execution-variant dspark --companion draft=/models/Qwen3.8-27B-DSpark \
  --max-sequence-length 512 -o qwen38-dspark.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-NVFP4 --precision fp16 \
  --execution-variant dflash --companion draft=/models/Qwen3.8-27B-DFlash2 \
  --max-sequence-length 512 -o qwen38-dflash2.bundle
```

For the additional mapped profiles, use a compatible prepared checkpoint:

```sh
trtmc qwen3_8 build /models/Qwen3.6-27B-NVFP4 --precision fp16 \
  --quantization nvfp4 --execution-variant mtp \
  --max-sequence-length 4096 -o qwen36-mtp.bundle

trtmc qwen3_8 build /models/Qwen3.8-27B-INT4-AWQ --precision fp16 \
  --quantization int4_awq --execution-variant dflash \
  --companion draft=/models/Qwen3.8-27B-DFlash2 \
  --max-sequence-length 512 -o qwen38-int4-dflash2.bundle
```

MTP uses the checkpoint's embedded draft, with three draft steps and four
verification positions; it takes no `--companion`. Its INT4-AWQ variant uses
`--quantization int4_awq` and a matching checkpoint. These examples describe the
mapped interface, not additional locally executed model tests.

DFlash2 uses Edge's public `dflash` mode. Paired builds invoke the official
ONNX exporter and native engine builder for both engines. They include required
embedding, selector/head, tokenizer, and prompt-template assets. Runtime
verification uses size 8, drafting top-K 1/step 1, and DSpark scheduling off;
the draft architecture selects the appropriate block semantics. DFlash2 DDTree
is not supported. The mapped DSpark context range is 9–1024 tokens; DFlash2 is
9–4096, also bounded by source metadata.

Ordinary builds map the official Python builder and native C++ runtime. TP2
uses one process with Edge-managed rank threads and requires two matching GPUs
and compatible NCCL. Mixed-NVFP4 uses `--dense auto`, preserving the checkpoint's
actual per-layer and KV-scale metadata instead of requantizing or forcing a
precision. Image profiles allow 128–1280 tokens per image and size the total
visual capacity to the LLM input budget. Edge owns pixel preprocessing.

Self-contained bundles include only original checkpoint shards named by emitted
runtime metadata, plus their unchanged index. Use scratch output and `TMPDIR`:
publication and extraction temporarily coexist with large bundles. Keep source
weights immutable through build completion.

Build preparation failures warn and attempt the unchanged native request once.
Native rejection remains explicit for unmapped combinations; a
failed pair never becomes ordinary base-only decoding. Runtime errors propagate
without fallback. Unsupported controls are rejected before execution. Accuracy
failures are evidence, **not** a dispatch denylist.

## Native fallback

The ordinary text fallback uses the family's TensorRT Python graph and native
C++ runtime, not ONNX or an Edge executable. Prepared ModelOpt NVFP4 metadata
selects the existing native quantization path even when `--quantization` is
omitted. Both mixed FP8/NVFP4 and uniform NVFP4 projections retain their packed
weights, source scales, and activation Q/DQ operations, including prepared MTP
draft projections. Actual exported LLM K/V scale tensors enable native E4M3
cache storage. The target, draft and verification graphs use the source scales
for Q/DQ and retain one-byte cache inputs/outputs; DeltaNet recurrent state stays
FP32. Without exported LLM scales, cache storage stays in the selected compute
precision. Broad quantization metadata alone does not enable FP8 KV: the locally
tested mixed checkpoint has no such scale tensors.

The FP8-cache path depends on the separate, model-agnostic runtime E4M3 dtype
mapping. A small prepared AWQ target and MTP build and run on SM89 with FP8 cache
I/O. Six greedy/sampled cases match the identical target, and six teacher-forced
K/V positions match independent E4M3 quantization byte-for-byte. These controls
do not qualify a full checkpoint or assert FP8 support on every GPU.

An ordinary request does not enable speculative execution merely because its
checkpoint contains draft weights. `--execution-variant mtp` maps to the native
draft and verification graphs with three draft tokens and four verification
positions. The ordinary native `--mtp-seq-len N` option keeps its requested count. That mapping and uniform projection loading have contract coverage,
not full uniform-checkpoint MTP model qualification.

The existing chunked-prefill implementation remains available for compatible TP1
text requests, including ordinary embedded-MTP requests. Explicit
`--prefill-chunk-len` selects native execution rather than dropping the option
in Edge offload. Native media, external feature drafts, TP2, exported FP8 KV,
and prepared MTP projections retain per-token prefill because the existing
chunked graph does not expose their input/state contract. These modes accept
an omitted option or zero; an explicit positive chunk size fails clearly.

There is no native GPU-SKU allowlist. TensorRT owns graph lowering. A direct
TensorRT 11.1 FP4 Q/DQ numerical control passed on SM89 and SM120; SM89 emitted a
warning about unaccelerated FP4 execution. The same control on SM80 was rejected
because its FP8 Q/DQ operations require FP8 support. These operator controls do
not qualify a full model, other TensorRT releases, or other operating systems.
The Edge route tables above describe separate offload profiles; they must not be
interpreted as intrinsic restrictions on the native model.

For the mixed-NVFP4 revision listed below, native FP16/context-256 ordinary
build and C++ execution passed on SM120 without Edge installed. The existing
capital-chat workload matched a fresh official-source FP32 oracle decoded from
the same checkpoint: identical generated token IDs, NED 0.0 at the unchanged
0.15 gate. The oracle does not emulate compiled activation quantization.
This one workload does not qualify raw-text, media, speculative, TP, or other
platform configurations.

The native reader also maps prepared ModelOpt `W4A16_AWQ` group-128 target
and embedded-draft projections. It preserves signed INT4 codes, per-group
weight scales, each projection's input smoothing, and Q/gate row interleaving;
it does not calibrate weights. A small native hybrid target built and executed
through the C++ runtime on SM80. This is integration evidence, not a 27B
checkpoint quality result.

New native MTP bundles use three sequential single-step draft calls with
isolated speculative KV state, followed by four-position target verification.
Previously built composed-draft bundles remain readable. This avoids the
TensorRT 11.1 tactic rejection observed in the composed INT4 draft graph without
decoding weights to FP16 or reducing the draft count.

The native verifier uses the requested target sampler, including temperature,
top-K/top-P, seed, repetition history, and EOS. Only confirmed target samples
advance the request history. Draft catch-up uses the preceding target hidden
state, and read-only verifier bindings preserve the ordinary decoder's position
inputs. Output-budget termination never commits an unconfirmed recurrent suffix.

A small prepared hybrid AWQ target and embedded draft build and execute through
Model Connect on SM80. Six controlled greedy/sampled requests, including the
context boundary, match the same native target token-for-token and are repeatable
with a fixed seed. These are integration controls, not full 27B MTP qualification.

Native embedded MTP also retains image, ordered-image, video and mixed-media
inputs. Teacher-forced draft prefill receives the target's visual rows and
three-axis rotary positions; generated draft/verification tokens use the
provider's logical decode offset without changing physical cache indices.
The C++ scheduler clears borrowed media state on success and error. A small
prepared-AWQ media bundle passed 19 typed greedy/sampled/error cases, each run
with ordinary same-target decoding and two MTP executions. Tokens match exactly
and seeded execution is repeatable. The same 19-case, three-mode control also
passes with a small uniform-NVFP4 target and embedded draft on SM89, including
its unaccelerated FP4 lowering. Both controls were repeated after the native
video-format correction below. They do not qualify a full prepared 27B MTP
checkpoint. External DSpark/DFlash2 remain text-only.

Native DSpark uses a family-owned TensorRT draft graph, original Markov and
confidence heads, and the target's exact FP16 embedding table and quantized LM
head. Seven bidirectional proposal rows are conditioned on confirmed target
features. C++ verifies proposals sequentially with `min(1, p/q)` acceptance and
positive-residual sampling, retaining the existing temperature, top-K/top-P,
min-P and repetition controls. Rejected proposals never enter recurrent target
state. This correctness-first implementation does not claim a speedup.

The retained mixed-NVFP4 27B target and DSpark companion built and executed
natively at FP16/TP1/context-256 on SM120. The existing capital-chat case matched
the cached independent FP32 reference exactly, including token IDs (NED 0.0),
with actual drafting and acceptance recorded. Small actual MC integration tests
also cover greedy target parity, seeded sampling repeatability, EOS and capacity.
This does not replace the known raw-text accuracy failure or establish broader
prompt, benchmark, context-size, or platform qualification.

Native DFlash2 uses an eight-row bidirectional draft with grouped dynamic
convolution, a sliding attention window, and seven prediction rows. The inherited
target LM head retains its exact quantization. A separate TensorRT selector uses
the original predecessor/successor codebooks and FP32 conditional scoring over
sixteen candidates. C++ proposal sampling and target verification preserve the
requested sampling controls; rejected proposals never enter target state.

The retained DFlash2 companion built natively at FP16/TP1/context-256 on SM120,
using the already-validated mixed-NVFP4 target engine and identical target feature
layers. Actual MC inference matched the same independent capital-chat reference
exactly (NED 0.0), with drafting and acceptance recorded. Small prepared AWQ
integration also covers greedy target parity, seeded sampling, EOS and capacity.
These results do not qualify a full AWQ checkpoint, broader quality workloads,
or every platform. Known accuracy issues remain runnable and are not overridden.

Native media bundles include a TensorRT visual tower, media-aware decoder, and
family-owned C++ RGB preprocessing, ordered image/video handling and three-axis
MRoPE. Runtime execution does not require Python or Edge. Prompt recipes are
derived from the source tokenizer during build; 44 C++ formatting comparisons
match the Qwen3.6 and Qwen3.8 provider templates byte-for-byte. A small real
prepared-AWQ bundle passed 11 typed execution/error-recovery cases, including
mixed image/video input. These component checks are not full-model quality evidence.

The retained mixed-FP8/NVFP4 Qwen3.8-27B checkpoint built and ran natively at
FP16/TP1/context-2048 on SM120 after correcting two source rounding boundaries:
activation reconstruction rounds its global scale to the compute dtype, while
weight reconstruction stays in FP32 until the complete weight is cast. Packed
weights, calibrated scales and requested quantization remain unchanged. A real
checkpoint slice matches the pinned source activation, weight and matrix
arithmetic exactly; this component check is not full-model quality evidence.

| Native mixed-quantized workload | Result against the original FP32 reference |
| --- | --- |
| Raw text | Accuracy issue, NED 0.21875 |
| Capital chat | Passed, NED 0.00000 |
| Single image | Accuracy issue, NED 0.16187 |
| Ordered two images | Accuracy issue, NED 0.78879 |
| Two-frame video | Accuracy issue, NED 0.62542 |
| Image and video together | Accuracy issue, NED 0.68707 |

All six requests completed. The existing NED 0.15 gate and original references
are unchanged. Accuracy failures remain runnable; residual numerical causes
are not localized or established TensorRT compiler defects. The corrected
target also passed the scoped DSpark and DFlash2 capital-chat comparisons.
The uniform-NVFP4 MTP integration control passed 19 text/media/error cases
with exact same-target tokens and repeatability; it does not qualify a full
prepared MTP checkpoint. No other platform or workload qualification is implied.

Original FP16 ordinary TP2 has a native path. It partitions full-attention and
DeltaNet heads and MLP channels; native TensorRT collective layers reduce the
row-parallel projections before residual additions. Norms, embeddings and the
vocabulary projection are replicated. Rank-zero broadcasts provide identical
media embeddings and sampling logits on both ranks. The builder compiles each
rank on its corresponding visible GPU; the C++ family owns two rank threads,
communicator lifetime, sharded state and error recovery. A compatible NCCL
runtime must be discoverable by the platform loader. No MPI launcher, family
CUDA kernel, or shared-core model dispatch is added.

Small actual FP16 TP2 graph tests agree with the native TP1 target within the
existing 0.001 relative/absolute diagnostic tolerance; rank logits are identical.
The public build and MC C++ media task pass 11 tiny text/image/video/mixed-media
and invalid-input recovery cases, with exact TP1/TP2 token agreement. These are
integration checks, not full-checkpoint TP2 quality or performance claims.

Both full original checkpoints built and executed natively at
FP16/TP2/context-4096 on two SM120 GPUs. These are scoped workload results,
not catalog-wide or registered-suite qualification:

| Native workload | Qwen3.6-27B | Qwen3.8-27B |
| --- | --- | --- |
| Capital chat | Passed, NED 0.0 | Passed, NED 0.0 |
| Raw text | Not rerun; the known markup issue remains recorded | Passed, NED 0.0 |
| Single image | Accuracy issue, NED 0.23063 | Passed, NED 0.07067 |
| Ordered two images | Accuracy issue, NED 0.51163 | Passed, NED 0.0 |
| Two-frame video | Accuracy issue, NED 0.57282 | Passed, NED 0.0 |
| Image and video together | Accuracy issue, NED 0.39492 | Passed, NED 0.0 |

Native video expansion retains the source processor's outer video wrapper,
per-temporal-group timestamped spans, and video-specific placeholder tokens.
The earlier native expansion copied Edge 0.11's flat image-pad expansion and
omitted two wrapper tokens. Correcting only native runtime formatting resolved
the measured Qwen3.8 video and mixed-media failures without rebuilding engines
or changing the original FP32 references or NED 0.15 gate. The Qwen3.6 cases
improved but still fail that gate and remain runnable.

Ten actual source-processor image, video, mixed, ordered-mixed, and odd-frame
prompts agree exactly with native C++ text, token IDs, three-axis rotary indices,
and decode offsets. The source oracle uses Transformers 5.14.1, also pinned by
Edge 0.11. Edge's upstream runtime is unchanged; its earlier results below are
not replaced by native results. Remaining numerical differences are not, by
themselves, evidence of a TensorRT compiler defect.

Remaining full-checkpoint qualification gaps include prepared MTP,
AWQ-DFlash2, and actual exported FP8 KV scales. Their native implementations
have small actual graph/runtime controls, not full prepared-checkpoint accuracy
results; no new calibration was performed. Broader workloads, contexts, and
platforms remain unqualified. Unsupported requests fail explicitly rather than
becoming ordinary text or a different parallelism.

## Edge evidence and remaining limits

All comparisons retain the owning NED **0.15** gate. Exact agreement with direct
Edge is transport parity, not an independent quality result.

| Exact workload | Build/inference | Independent quality |
| --- | --- | --- |
| Qwen3.6 original FP16 TP2, raw text, 16 greedy tokens | Model Connect and Edge passed, exact agreement | NED 0.0, but the unchanged no-thinking-markup predicate failed; the FP32 reference also emits the markup |
| Qwen3.6 original FP16 TP2, capital chat, 10 tokens | Model Connect passed | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.6 original FP16 TP2, single image, 128 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.23063 |
| Qwen3.6 original FP16 TP2, ordered two images, 128 tokens | Both passed, exact agreement | NED 0.06217 passes, but the required giant-panda content check fails |
| Qwen3.6 original FP16 TP2, two-frame video transition, 64 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.63754 |
| Qwen3.8 original FP16 TP2, same raw text | Both passed, exact agreement | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, capital chat, 10 tokens | Model Connect passed | Fresh FP32 comparison passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, single image, 128 tokens | Both passed, exact agreement | Fresh FP32 comparison and woman/dog/beach content checks passed, NED 0.0 |
| Qwen3.8 original FP16 TP2, ordered two images, 128 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.19279 |
| Qwen3.8 original FP16 TP2, two-frame video transition, 64 tokens | Both passed, exact agreement | Failed fresh FP32 comparison, NED 0.62058 |
| Qwen3.8 mixed NVFP4, ordinary/DSpark/DFlash2, raw text, 16 tokens | All three passed, exact direct agreement | All failed NED 0.21875 against the same checkpoint decoded to FP32 |
| Qwen3.8 mixed NVFP4 ordinary, single/ordered images | Both passed, exact direct agreement | Failed matched packed-weight FP32 comparisons: NED 0.27907 / 0.77639; ordered-image giant-panda content check also failed |
| Qwen3.8 mixed NVFP4 ordinary, two-frame video transition | Both passed, exact direct agreement | Failed matched packed-weight FP32 comparison, NED 0.70667 |

The mixed-NVFP4 ordinary, DSpark, and DFlash2 capital-chat requests passed fresh same-checkpoint
FP32 comparisons (NED 0.0, 10-token budget). This does not override their raw-text
failure or establish the registered original-checkpoint E2E/benchmark result.
The FP32 mixed-weight oracle explicitly decodes original ModelOpt weights; it
does not emulate compiled activation or KV quantization. A difference is not,
by itself, evidence of a TensorRT compiler bug.

Empty-video rejection and a successful image request afterward were tested.
Video evidence currently covers a synthetic two-frame animal transition, not
general motion understanding. No quality criterion was relaxed.

Tested source revisions:

- Qwen3.6 original: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Qwen3.8 original: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Mixed NVFP4: `319f741cce68d7914884900c138a1fbb70a42f30`.
- DSpark: `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
- DFlash2: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

Original-weight DFlash2 remains outside this adapter's current admission and
has no executed result. Its initial admission decision used a 48 GiB device
budget and roughly 57.6 GB of FP16 text/draft source weights, not an observed
GPU OOM or measured engine allocation. Larger-memory devices are now available,
so that historical budget is not a current hardware blocker or an intrinsic
model restriction. The pinned ONNX builder rejects speculative TP greater than
one; neither that constraint nor native integration controls qualify this
currently unmapped profile.

The additional INT4 DFlash2 and uniform NVFP4/INT4-AWQ MTP mappings have not
been validated with full prepared checkpoints. Their Edge mappings delegate to
the documented upstream flows; small native engine/runtime controls described
above do not replace those missing full-model results. Reproducing every upstream calibration recipe is not a
prerequisite for using an already-compatible checkpoint or publishing this
integration. The original media builds retain unchanged FP32/content gates.
DSpark remains mixed-NVFP4-only; uniform
INT4-AWQ is admitted only for DFlash2 and MTP, never ordinary execution.
These results do not claim catalog-wide or registered CI coverage.

The existing original-checkpoint premerge manifests use FP16 TP1, not the TP2
profiles above. Their full E2E runs and registered MMLU/performance benchmarks
remain unexecuted; the capital-chat comparison is not a replacement for those
suites. The former 48 GiB hardware survey does not establish a current blocker
for those unexecuted single-device profiles; they must not be counted as executed
Edge failures or as qualified by the TP2 results.

The official performance tables also contain batch-8 MTP rows. This adapter
currently maps only batch 1; it does not implement or claim the batched profile.
Other published platform rows are not qualified by the recorded SM80/SM120
results.
