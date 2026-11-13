# Qwen3.5: optional Edge-LLM 0.11 execution

This family owns Edge selection, checkpoint admission, build arguments, bundles,
runtime orchestration and validation. Model-specific dispatch does not change
shared Model Connect code.
The native SDK is pinned to the official public Edge-LLM **0.11.0** revision
`95515c2f87fba8982db5a519f9022277667b3cc9`. Build on the executing GPU; there is
no cross-compilation or alternate vendor checkout.

## Installation and ownership

Enable `TRTMC_ENABLE_EDGELLM=ON` during native CMake provisioning and expose the
installed SDK through `CMAKE_PREFIX_PATH`. The adapter verifies the release,
architecture, GPU target, CUDA and TensorRT identities before offloading.

Compatible installed Python tools, including `tensorrt-edgellm==0.11.0` from
PyPI, are discovered automatically. Otherwise the adapter uses the SDK's private
source-installed Python. The wheel does not replace the native C++ SDK headers
and archives. There is no model-time installation or multi-version provider.

`edge_llm/builder.py` calls the official Python builder
(`experimental.builder.cli`); `runtime/edge_llm/` calls the native Edge runtime.
Edge owns the complete network, quantized operations and speculative execution.
The family does not implement replacement kernels or modify vendor sources.

## Execution scope

All Edge routes require FP16 compute, batch one, CP one and a matching native
Linux x86 SDK. Original sources use SM80 or SM120; admitted ModelOpt formats
use SM120. Checkpoint-driven weight precision is separate from compute precision.

| Configuration | Owning route |
| --- | --- |
| Dense 0.8B, 2B, 4B and 9B, Instruct and Base | Original-weight text and image generation, TP1 |
| Dense 27B | Original-weight text/image, TP2 LLM plus TP1 visual component |
| Embedded MTP, original 0.8B/2B/4B/9B topologies | Text/image, TP1; three draft steps |
| Embedded MTP, ModelOpt NVFP4 0.8B/2B/4B/9B topologies | Text/image, TP1; local model execution evidence is 4B only |
| Listed 4B/9B DFlash companions | Original matched target/draft; linear or DDTree, block16 |
| Listed 4B NVFP4 DFlash pair | Matched official target/draft exports, linear or DDTree |
| Dense standalone ModelOpt targets | NVFP4, FP8, MXFP8, INT4-AWQ group128, INT8 SmoothQuant; TP1 |
| 2B FP8 weights with calibrated FP8 KV | Standalone text only, TP1 |
| 2B NVFP4 weights with NVFP4 LM head | Standalone text, checkpoint-driven component precision |

This table describes execution admission, **not universal quality qualification**.
NVFP4 MTP uses the same four topology admissions as original-weight MTP. Edge
offload model-engine evidence covers the 4B NVFP4 profile; the native fallback
also has the 2B NVFP4 execution and accuracy limitations detailed below. The
other NVFP4 sizes are command mappings, not locally executed or qualified results.
Results belong to individual checkpoints, profiles and requests. Accuracy-only
failures remain executable; they do not become passing results.

Qwen3 and older, MoE, Qwen3.6/3.8, unknown quantization metadata,
unsupported component combinations, and speculative TP2 are outside this owner.
The separate Qwen owners must establish their own support.

## Test scheduling

The existing 2B TP1 image case remains in premerge and accepts either Edge
or the family-owned native media bundle. Its pinned checkpoint, image, prompt,
FP32 reference, expected answer and quality threshold are unchanged. The
legacy case ID retains `edge` for compatibility; backend admission is no
longer Edge-only. The 27B TP2 case remains explicit validation outside premerge
because it requires multiple GPUs. Existing 0.8B, 2B, 4B and 9B text coverage
also remains in premerge with native fallback when Edge is unavailable.

## Build and run

The existing family CLI protocol owns these arguments. No shared parser hook,
global variant option or core model dispatch is added.

```bash
trtmc qwen3_5 build /models/Qwen3.5-2B \
  --precision fp16 --max-sequence-length 256 -o qwen35.bundle
trtmc run qwen35.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

Image bundles use `--task vision_language_generation`. The Model Connect request
accepts `--image /path/to/image` or an ordered JSON `--images` list, not both.
A vision tower and source processor metadata are required. Edge-backed execution
preserves the provider chat template and delegates image preprocessing to Edge.
Native fallback instead uses the family-owned C++ preprocessing and checkpoint
chat-template handling; it does not invoke an Edge runtime.

```bash
trtmc qwen3_5 build /models/Qwen3.5-4B \
  --precision fp16 --task vision_language_generation \
  --max-sequence-length 8192 -o qwen35-image.bundle
trtmc run qwen35-image.bundle --image /images/dog.jpg \
  --prompt "What animal is shown?" --max-new-tokens 16 \
  --temperature 0 --top-k 1 --top-p 1 \
  --use-chat-template true --enable-thinking false
```

The source image-token profile is 128–1280 per image, bounded by the aggregate
visual capacity. Image examples do not qualify arbitrary resolutions.

### Typed video requests

Visual bundles also expose the existing native `VideoTextToText` and
`ImageVideoTextToText` APIs. Pass decoded, ordered RGB frames in `VideoView`;
frames must share dimensions. Optional timestamps must be finite, nondecreasing
and match the frame count. Without timestamps, the owning Edge mapping uses
one frame per second. The adapter preserves video identity and mixed-content
order rather than converting a video into independent image requests.

Build the same `vision_language_generation` bundle shown above. Use the typed
C++/C API for video input; the generic `trtmc run --image` option is not a video
interface, and no shared CLI option is added. Text-only bundles do not expose
the video tasks. Edge-backed execution delegates preprocessing, temporal patches
and the provider template to Edge. Native fallback uses the family-owned C++
preprocessing, temporal patching and template mapping instead. Existing visual
and context capacity limits apply to both routes.

Local evidence: original Qwen3.5-2B, FP16, TP1, context8192, two identical frames
from the existing car fixture. Model Connect and direct Edge returned identical
text, `white`; independent FP32 HF comparison passed the unchanged NED0.15
gate with NED0.0. This validates video transport and appearance, not motion
understanding, every checkpoint, or speculative video accuracy. Native owning
regressions additionally check distinct-frame ordering, mixed image/video
messages, timestamp preservation and malformed-input rejection.

### Speculative execution

```bash
trtmc qwen3_5 build /models/Qwen3.5-4B --precision fp16 \
  --execution-variant dflash --companion draft=/models/Qwen3.5-4B-DFlash \
  --max-sequence-length 2048 -o qwen35-dflash.bundle

trtmc qwen3_5 build /models/Qwen3.5-4B --precision fp16 \
  --execution-variant dflash_ddtree --companion draft=/models/Qwen3.5-4B-DFlash \
  --max-sequence-length 2048 -o qwen35-ddtree.bundle

trtmc qwen3_5 build /models/Qwen3.5-2B --precision fp16 \
  --execution-variant mtp --max-sequence-length 2048 -o qwen35-mtp.bundle
```

MTP reads the embedded draft layer and rejects `--companion`. DFlash checks the
draft architecture, vocabulary/hidden widths, target layers, mask token, context
and matching weight format. Linear uses top-k one; DDTree uses top-k four.
Text speculative builds cap input at min(context, 1024) while retaining the
requested KV capacity. Runtime sampling qualification remains greedy.

Python callers can use the existing family extension
`with_execution(request, BuildExecutionInputs(...))` from
`families.qwen3_5.edge_llm.config`. CLI help works offline.

### Tensor parallelism

```bash
CUDA_VISIBLE_DEVICES=0,1 trtmc qwen3_5 build /models/Qwen3.5-27B \
  --precision fp16 --tensor-parallel-size 2 --max-sequence-length 4096 \
  -o qwen35-27b.bundle
CUDA_VISIBLE_DEVICES=0,1 trtmc run qwen35-27b.bundle \
  --prompt "The capital of France is" --max-new-tokens 16 \
  --temperature 0 --top-k 1 --top-p 1 --use-chat-template false
```

Both devices must match the bundle target. A compatible NCCL runtime must be
discoverable through loader paths or `EDGELLM_NCCL_SO_PATH`. The family delegates
rank orchestration to Edge's `ParallelExecutionConfig`; it does not launch a
second Model Connect rank coordinator.

TP2 visual builds invoke the LLM and visual components separately because the
pinned combined builder does not accept that composition. Runtime receives both.
Speculation plus TP2 is not admitted. The 27B single-device DFlash pair exceeded
the compatible capacity used for validation; ordinary TP2 results do not qualify it.

### Quantized checkpoints

Use official Edge 0.11 tools and their declared dependencies, including
Transformers 5.14.1, to prepare quantized sources. The adapter consumes the
resulting checkpoint; it does not calibrate, repack or silently dequantize it.
Embedded and sidecar algorithms must agree. The builder uses `--dense auto`
for quantized sources, preserving projection exclusions and component metadata.

For FP8 KV, both the sidecar FP8 algorithm and the embedded static eight-bit
float KV scheme are required. Only the verified 2B standalone text combination
is admitted. The official draft exporter may declare NVFP4 only in its ModelOpt
sidecar; this specific draft schema is also supported.

## Validation boundaries and known quality issues

Release-specific local validation covers the named original dense checkpoints,
recorded image requests, MTP/DFlash modes and tracked standalone quantization
profiles. It includes actual Model Connect builds/inference, direct Edge controls,
existing owning comparisons and selected installed-package replays. Older
0.10 results are not evidence for this release.

The existing short-reference normalized edit-distance gate remains **0.15**.
The public 128-token basic fixture retains ROUGE-1 **0.25** and ROUGE-L **0.20**.
Generated token IDs must be present, valid and consistent with returned text.
Numeric scores alone do not certify factual correctness.

- Original small-model text profiles and original 4B/9B speculative text
  profiles pass their recorded short comparisons. Some termination-token
  suffixes differ from Hugging Face; decoded agreement is not exact EOS parity.
- Original 27B TP2 raw text and short single-image requests pass. Its longer
  two-image request remains unqualified: Edge misses one image's subject and
  the required distributed FP32 reference also produced invalid repetitive text.
  A different-precision diagnostic is not a substitute for that reference.
- Quantized and some Base multi-image profiles fail independent-reference NED
  despite matching direct Edge. Their execution remains available.
- Several small-model quantized long answers contain incorrect biographies,
  founders or dates despite passing numeric gates. The 2B FP8-KV direct output
  also invents NVIDIA's founders. An original unquantized 2B HF control has
  related factual errors, so this category is not established as Edge-only.
- The 0.8B INT4-AWQ short answer fails NED in both experimental and ONNX direct
  controls. Some 4B INT4-AWQ/INT8-SQ long outputs also fail semantic review.
  ONNX controls are diagnostics, not an integrated Model Connect ONNX route.

These observations are not an exhaustive benchmark, stochastic-generation,
full-context or cross-platform qualification. Existing quality gates and thresholds
are unchanged; the family does not skip a configuration because its quality
comparison fails.

## Native fallback implementation status

Native TensorRT graph construction now reads already-prepared ModelOpt NVFP4
and FP8 projections without dequantizing their stored weights or recalibrating.
It preserves source-excluded dense projections, interleaved query/gate rows,
calibrated activation scales and a prepared LM head. DeltaNet recurrent state
remains FP32. Explicit `--quantization` is family-owned and must match the source;
it never requests an implicit format conversion.

A prepared Qwen3.5-2B NVFP4 checkpoint with an NVFP4 LM head built through the
public Model Connect build command and executed through the native C++ runtime
on SM89, using FP16 compute, TP1 and context256. The existing ten-token capital
question matched the retained original FP32 reference: NED 0.0 against the
unchanged 0.15 gate. This is one scoped native result, not quantized-HF exact
parity, long-answer accuracy, all-platform qualification or validation of every
size in the Edge admission table.

Small actual NVFP4 and FP8 projection controls also compare against the pinned
release's independent eager arithmetic. The NVFP4 graph retains FP32 global
scales for dynamic quantization and the source's FP16-rounded global scale for
activation reconstruction; those are distinct precision boundaries.

Prepared FP8 checkpoints with exported LLM K/V scales use exact E4M3 cache
storage in the native graph and C++ runtime. Current and cached entries use the
original calibrated scales; DeltaNet recurrence remains FP32. The retained 2B
FP8/KV checkpoint passed an actual native build and C++ run against the existing
short original-FP32 reference. Byte-level controls cover every exported
attention-layer scale pair. This does not qualify longer generation or media.

ModelOpt INT4-AWQ, INT8 SmoothQuant and MXFP8 native text projections also
preserve their original packing, activation smoothing and scale types. Small
AWQ controls match the source arithmetic exactly. MXFP8 matches the pinned
production lowering, but not its separate eager activation reference. A small
SmoothQuant difference is localized to fused INT8 rescaling versus FP16 eager
intermediates. These diagnostics do not qualify full checkpoints or change any
existing quality criterion. Full-model validation of the recorded prepared
size/format combinations is blocked on unavailable exported tensors; component
controls do not establish full-checkpoint qualification.

The native media path includes a family-owned TensorRT visual graph and C++
image/video preprocessing, provider-template formatting, multimodal rotary
positions and typed image, video and mixed-media task bindings. It does not use
ONNX or an Edge runtime. Original Qwen3.5-2B, FP16, TP1, context8192 was built
through the public build command and exercised on SM89:

| Scoped request | Native execution | NED against original FP32 |
| --- | --- | --- |
| Single image, 16-token limit | Pass | 0.000, pass |
| Two ordered images, 128-token limit | Pass | 0.058, pass |
| Video, 64-token limit | Pass | 0.502, accuracy issue |
| Image plus video, 128-token limit | Pass | 0.289, accuracy issue |

The unchanged gate is 0.15. Both accuracy failures remain runnable. A private
boundary check matches source prompt text, tokenizer IDs and multimodal rotary
positions for the failed cases' source geometry; it does not establish visual
feature or decoder-logit equality. The accuracy differences are not yet
attributed to an Edge or TensorRT defect. These results do not qualify Base
checkpoints, other model sizes, prepared media formats or longer contexts.

Native embedded MTP builds the trained draft head, target and four-token
verifier, preserving three draft steps rather than substituting ordinary
execution. Draft candidates use isolated scratch KV; only confirmed target
history updates persistent state. Sampling, EOS, output budgets and media
rotary positions remain family-owned.

The original 2B checkpoint, FP16/TP1/context8192, built through the public
`qwen3_5 build --execution-variant mtp --task vision_language_generation`
command on SM120 after the missing optional SDK triggered native fallback.
Actual C++ typed text, single-image, ordered-image, video and mixed-media
requests matched an ordinary control using the **same target engine bytes**.
All ten greedy/seeded-sampling comparisons and repeated MTP requests produced
identical tokens. Execution exercised full acceptance and partial recovery;
this is not a speedup claim.

Against the independent original FP32 reference and unchanged NED0.15 gate:

| Scoped MTP request | Native execution | NED against original FP32 |
| --- | --- | --- |
| Capital question, 10-token limit | Pass | 0.000, pass |
| Single image, 16-token limit | Pass | 0.000, pass |
| Two ordered images, 128-token limit | Pass | 0.396, accuracy issue |
| Video, 64-token limit | Pass | 0.502, accuracy issue |
| Image plus video, 128-token limit | Pass | 0.289, accuracy issue |

The same three accuracy failures occur with speculation disabled in the
same-target control, and all remain runnable. The ordered-image result is a
separate SM120 typed-RGB experiment from the earlier SM89 CLI image result;
no GPU-only or MTP-only cause has been established.

The original 4B checkpoint, FP16/TP1/context4096, also built and completed all
ordinary, MTP and repeated-MTP requests. Text, single-image and ordered-image
quality passed (NED 0.0, 0.0 and 0.096); video and mixed media failed the unchanged
gate in both ordinary and MTP modes. Nine of ten greedy/sampled MTP requests
matched the same-engine ordinary target exactly; one greedy mixed-media output
differed, although all ten seeded repeats were stable. This is an unresolved
native target-parity issue, not an execution failure or a proven Edge/TRT defect.
The configuration remains runnable but is not qualified for exact target parity.

The retained full 2B NVFP4 checkpoint, including its packed LM head and embedded
draft, also builds and executes MTP on SM120 with FP16 compute, TP1 and
context8192. Text and single-image quality pass; ordered images, video and mixed
media fail the unchanged gate (NED 0.724, 0.608 and 0.619). All remain runnable.
Only six of ten greedy/sampled requests exactly match the same-engine ordinary
control, although all ten MTP repeats are deterministic. This prepared profile
is **not qualified for exact target-token parity**. A private same-state probe
localizes differences to later verifier rows; changing temporary KV layout did
not fix them and that candidate was not adopted. The cause is not established
as an upstream TensorRT or Edge defect. These results do not qualify other
sizes, Base checkpoints, prepared exports or additional platforms.

The original 0.8B Instruct checkpoint also built and ran ordinary and MTP
requests at FP16/TP1/context4096. Short text, single-image and video comparisons
passed with NED0.0; ordered images passed with NED0.010. Mixed media failed the
unchanged gate in both ordinary and MTP modes (NED0.338 and NED0.333).
Nine of ten greedy/seeded MTP comparisons matched the same native target;
greedy mixed media first diverged at generated-token index 32. These results
are request-scoped, not full-size or prepared-checkpoint qualification.

The native tokenizer loader merges source `added_tokens_decoder` metadata from
`tokenizer_config.json` with `tokenizer.json`, preserving token IDs and behavior.
Original Base checkpoints define their thinking suffix tokens only in the former.
Ignoring those definitions split the suffix into ordinary tokens and shifted
multimodal positions. This was a Model Connect input bug, not an established
Edge or TensorRT defect. Earlier Base findings from that incorrect input path
are superseded by the corrected-runtime results below.

All four original Base checkpoints completed native ordinary, MTP and repeated-MTP
requests at FP16/TP1/context4096. The 4B and 9B replays reused the exact successful
pre-fix engines; the 0.8B and 2B engines were rebuilt because their earlier engine
files had been retired. All comparisons reused the saved original FP32 references
and unchanged quality gates. Ordinary controls share the MTP bundle's target and
visual engine bytes; they are not separate ordinary-builder qualifications.

| Base checkpoint | Post-fix ordinary and MTP quality | Exact MTP/ordinary tokens | MTP repeatability |
| --- | --- | --- | --- |
| 0.8B | Single image, video and mixed media pass; raw text and ordered images fail | 8/10 | 10/10 |
| 2B | All five recorded requests pass | 10/10 | 10/10 |
| 4B | All five recorded requests pass | 10/10 | 10/10 |
| 9B | Raw text, single image, video and mixed media pass; ordered images fail | 10/10 | 10/10 |

The five quality requests are raw text, single image, ordered images, video and
mixed image/video. Target-parity and repeatability counts additionally include
five seeded-sampling requests. The 2B and 4B mixed-media NEDs are 0.020 and 0.039;
their other recorded quality comparisons have NED0.0. The 9B ordered-image NED
is 0.206 in both modes. The 0.8B raw-text NED is 0.651; ordered-image NED is
0.307 ordinarily and 0.353 with MTP. Its greedy ordered-image tokens first differ
from the native target at index 100, and sampled video first differs at index 2.
These remaining accuracy and target-parity issues stay runnable and recorded;
the input fix does not establish an upstream cause for them. These are scoped
requests, not universal model, context, format or platform qualification.

Original 9B MTP at FP16/TP1/context4096 likewise completed its build and all
ordinary/MTP/repeated-MTP requests. Short/long text and single image passed
with NED0.0; ordered images passed with NED0.015. Video and mixed media failed
the unchanged gate in both modes. Eleven of twelve greedy/seeded comparisons
matched the same native target; greedy mixed media first differed at generated
token index 117. All twelve MTP repeats were stable. These are native accuracy
and target-parity findings, not build failures, memory failures or established
Edge-only defects. Exact-target controls reuse the same target/visual engine
bytes, and do not represent separate ordinary-builder qualification.

Native original-weight FP16 TP2 now partitions attention heads, DeltaNet heads
and MLP channels, with TensorRT collectives before replicated residuals. Both
ranks execute in one C++ process; no MPI launcher is required. Embeddings,
vocabulary projection and visual tower are replicated, with a visual plan
built for each rank device. NCCL supplies the runtime communicator. TP2 does
not silently replace requested quantization or speculative decoding.

A small hybrid text/media control matched TP1 tokens across raw/chat/thinking
text, single/ordered images, video and mixed media, including recovery after
three rejected requests on both SM80 and SM120. This establishes native
rank/task integration, not full-checkpoint quality. Full original 27B FP16 TP2,
context8192, builds both target and visual rank plans on SM120, but execution
context creation exceeds the tested pair of 32-GB GPUs. A subsequent native
build and C++ run on two 40-GB A100s succeeded. The single-image comparison
passed with NED0.0. Raw text matched the FP32 reference but failed the existing
thinking-disabled predicate; the ordered-image comparison failed against a
degenerate FP32 reference. These are recorded accuracy/oracle issues, not a
remaining memory failure. Reusing that engine, video and mixed-media requests
passed the unchanged FP32 comparison gate with NED0.000 and NED0.089. Both
video outputs incorrectly claimed that only one image was provided, so reference
agreement does not establish motion understanding. The earlier raw-text and
ordered-image issues remain unresolved. The native runtime accepts scalar or
multiple checkpoint EOS IDs and preserves an explicit request-level EOS override.

Native DFlash builds the actual companion graph and exports the ordered target
features it consumes. Linear decoding uses proposal acceptance and residual
rejection sampling; DDTree constructs scored branching proposals and supports
greedy decoding. Only confirmed target tokens update recurrent state, so rejected
branches cannot contaminate subsequent requests. This is not a speedup claim.

Original Qwen3.5-4B plus its original DFlash companion, FP16/TP1/context4096,
built both linear and DDTree modes through the public family command and ran
through the MC C++ task APIs. In each mode, four reference comparisons passed: short text, long text and one image
had NED0.0; ordered images had NED0.096 against the unchanged 0.15 gate. These
results do not qualify 9B or other prepared checkpoints. Both modes also execute
the full 4B video and mixed-media requests and produce identical tokens to each
other and to the ordinary native target, but fail the unchanged FP32-reference
gate: NED 0.669 and 0.738. Those
accuracy failures remain runnable; no Edge-only or TensorRT-only cause has
been established.
Original Qwen3.5-9B plus its original DFlash companion also completed native
FP16/TP1/context4096 linear and DDTree builds and six C++ requests per mode.
Short and long text and single-image comparisons passed with NED0.0; ordered
images passed with NED0.015. Video and mixed media executed but failed the
unchanged gate (NED0.704 and NED0.668). Both modes accepted real draft tokens
on every request and produced identical output tokens. All six outputs also
matched an ordinary control built from the linear bundle's exact target engine
bytes, so those failures are not specific to paired decoding. They remain
runnable; these results do not establish speedup or qualify prepared exports.

Small original and NVFP4 paired controls exercise both schedulers with actual
C++ inference, same-target greedy parity, sampling repeatability, request reset
and error recovery. Packed controls preserve distinct target/draft LM-head scales
and accept the official draft's sidecar-only ModelOpt metadata. They do not
qualify full prepared checkpoints whose exported weights are unavailable.
The metadata matcher also accepts a matching prepared 9B pair, but no complete
prepared target/draft was available for that experiment. The listed original 9B
results do not qualify this combination; it remains an explicit input-availability
limitation, not an observed Edge or TensorRT command failure.

**Native fallback completion remains in progress.** Exact size/format-specific
media, MTP and paired validation remain incomplete. Unsupported combinations
fail explicitly instead of falling back to ordinary text, unquantized weights
or different speculation. Existing Edge routes do not establish native coverage.

## Artifacts and failure behavior

The bundle includes the required engines, tokenizer/template metadata and
external-weight dependencies. TP2 retains the original shards named by Edge's
runtime identity, not another copy of every baked projection. Keep sources
immutable during building and budget for bundle staging and runtime extraction.

Ordinary requests without an optional SDK remain native without a warning.
Incompatible SDKs and Edge preparation failures preserve diagnostics, warn and
retry the unchanged native request once. A requested speculative mode must
never silently become a base-only bundle. Runtime failures do not switch backends.
Use scratch-backed output and `TMPDIR` paths for large artifacts.

Run the existing owning checks with:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/qwen3_5/tests -q
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/qwen3_5/tests/test_e2e.py --e2e-testcase qwen35-2b -q
```

Selected E2Es require a checkpoint and `TRTMC_BINARY`/`TRTMC_RUNTIME_ROOT`
pointing to the native runtime built for the executing GPU. Unselected E2E
skips are not validation evidence.
