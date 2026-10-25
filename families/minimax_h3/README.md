# MiniMax-H3 image Ref2VA: experimental integration

The image-reference path is implemented but **not qualified end to end** with
the complete trained model. Do not advertise it as production support,
quality-equivalent inference, or real-time generation. Automatic discovery in
`support.py` remains unchanged. A passing factory/host test does not execute or
qualify the eight inference components.

This path requires the original `transformer_ref` checkpoint partition. A
text-only checkpoint, including a text-only FastH3 checkpoint, is not a
replacement. The default profile provides image-reference conditioning;
the explicit two-window development profile below adds latent continuation.
Neither profile enables initial-frame editing or four-step reference inference.

## Explicit experimental build

Use a complete local original checkpoint. The trained component oracles pin
`MiniMaxAI/MiniMax-H3` revision
`48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc`; no download is implicit in the
example. Provide an existing output directory with space for the actual
serialized bundle and compiler scratch. This explicit Ref2VA path enables
single-file staging: it reserves a 64-KiB format-v1 JSON header, appends every
plan once and atomically publishes that same temporary file. It no longer
keeps a separate set of section files while copying the complete final bundle.
Single-file staging itself does not change plan/config/tokenizer bytes or
runtime settings. If replacing an existing destination, that previous file
still occupies space until atomic
publication. Bundle size, compiler scratch and host/GPU memory are separate
quantities; the measured example below does not
establish a minimum RAM/VRAM figure or resolve the failed trained-video
numerical/storage gates.

One complete original-checkpoint `minimal` build on a single GB300 produced a
128,526,853,374-byte bundle (119.7 GiB), including all eight plans, tokenizer
and runtime metadata. Its instrumented build, staging, payload readback and
hash-verification interval was 605.8 seconds; the builder process's maximum
RSS was 104,839,424 KiB (approximately 100 GiB). These are observations for
that exact build, not inference latency or minimum host RAM. Compiler scratch
and exact GPU allocation peaks remain unqualified. Container memory counters
also include page cache and prior activity; they are not model RAM minima.

With `core/builder` and the repository root on `PYTHONPATH`:

```python
import os
from pathlib import Path

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3.reference_build import build

request = BuildRequest(
    model_dir=Path(os.environ["H3_CHECKPOINT"]),
    output_path=Path(os.environ["H3_OUTPUT"]),
    family="minimax_h3",
    task="references_text_to_audio_video",
    precision="bf16",
    backend="trt",
    image_height=1344,
    image_width=768,
    video_num_frames=124,
)
writer = BundleWriter(request.output_path)
writer.record_build(request)
try:
    build(
        request, writer,
        memory_preset=os.environ.get("H3_MEMORY_PRESET", "minimal"),
        reference_capacity=os.environ.get("H3_REFERENCE_CAPACITY", "standard"),
    )
    writer.finish()
except BaseException:
    writer.abort()
    raise
```

The family-local entry point is intentional: it exposes the experimental
memory and reference-capacity presets without inventing shared CLI options or
changing discovery.
It validates original configs, tokenizer, key partitions and normalization
statistics. Selected weights support the original indexed components and the
original single-file, unindexed audio VAE; only requested tensors are loaded.
Ambiguous layouts and missing selected keys fail explicitly. It does not
establish model correctness merely by producing a
bundle. Supported build settings are batch 1, TP/CP 1 and no quantization;
unsupported overrides fail explicitly.

## Residency presets

| Preset | Module residency | Video tile batch |
| --- | --- | --- |
| `minimal` (default) | Staged; release completed phases | 1 |
| `original` | Resident modules/cache | 28 |

Both preserve original trained partitions, reference preprocessing, output
geometry, precision boundaries and 49 denoiser forwards. No resolution, frame
count, step count or model depth is reduced to save memory. However, neither
end-to-end preset equivalence nor cross-preset numerical identity is currently
qualified; changing batch size can change TensorRT tactics and numerical
results. Do not interpret the presets as a proven quality/speed tradeoff.

Plans are loaded one at a time from the bundle. The largest plan still incurs
a transient CPU read buffer. Minimal residency also does not remove the full
FP32 video result: default geometry alone needs 1,535,901,696 bytes for that
buffer. There is no qualified minimum RAM/VRAM figure for the complete path.

## Reference capacity profiles

`reference_capacity` is independent of `memory_preset`. It changes the language
and DiT operand profiles, not checkpoint partitions, image preparation, output
geometry, precision, frames, steps, or model topology.

| Capacity | Maximum text rows, including visual tokens | DiT condition-video min/opt/max rows |
| --- | --- | --- |
| `standard` (default) | 32,768 | 4,096 / 5,504 / 16,384 |
| `six_portrait` (experimental; one complete native request executed, not qualified) | 65,536 | 4,096 / 43,776 / 49,152 |

Original preparation maps a 9:16 image to 3648x2048. Each such image contributes
7,296 rows to **each** conditioning stream; six contribute 43,776 before prompt
and presentation tokens. The standard budget therefore cannot accept six
distinct portrait anchors. Reusing one exactly identical accepted tensor for
several semantic roles is different from discarding distinct references.

The expanded maximum also fits six images with aspect ratios up to 2:1, subject
to the separate text/presentation budget. It is a row budget, not a promise to
accept any six image shapes or any prompt length. The pipeline rejects either
overflow before loading engines; it never downsizes or omits references to fit.

Select the capacity **at build time** and use the matching newly built bundle.
Both the language and DiT plans require larger profiles. Editing old bundle
metadata or merely increasing a client's reference-count setting does not
expand those engines. Legacy default metadata remains unchanged; expanded
metadata has an explicit capacity label and all budgets must agree.

The expanded `minimal` profile was built on one GB300 from source
`6c83ec236c9d00ed1b0652cae1c275ff641d3f4c`, using the pinned original
checkpoint and BF16 build request with the unchanged 1344x768, 124-frame,
50-grid recipe. Its
complete eight-plan bundle is 128,527,457,708 bytes (119.7 GiB). Every section
and the complete retained bundle were SHA-256 verified; publication preserved
the single spool's inode. The instrumented build, staging, payload readback
and hash-verification interval was 614.4 seconds, and the builder process's
maximum RSS was 104,779,648 KiB (approximately 100 GiB). These are observations
for this expanded build, separate from the earlier standard-profile figures.

The matching ARM64 family DSO and public SDK/CLI also compiled at that
source revision. Both owning metadata/preflight CTests
passed with CUDA hidden. This is native binary and CPU-contract evidence,
not trained inference.

One complete six-image native request using that exact expanded bundle has
executed on a single GB300: seed 42, 1344x768, 124 frames at 24 fps and original
stereo 32-kHz audio. Its cold bridge load, generation and raw-output write took
1079.99 seconds; this is not isolated generation latency or warm throughput.
The experiment wrapper failed while writing its final report after native
generation completed. A separate read-only audit verified the complete retained
video/audio payloads; the failed wrapper record was not replaced with a pass.
The original reference implementation also executed with the same six images,
prompt and seed, but numerical equivalence and human quality acceptance remain
unqualified. This is one executed case, not general six-reference support.

The sampled device memory included pre-existing allocations, so isolated
inference RAM/VRAM peaks, minimum capacity and cross-preset fit remain
unqualified. Neither build RSS nor container memory counters establish minimum
RAM/VRAM requirements. Larger profiles can change tactics and memory use;
these observations do not waive the existing trained numerical, serialized
video-plan storage or quality gates.

## Bundle composition

The `minimax_h3.image_ref2va/v1` metadata schema identifies these eight plans:

| Plan | Role |
| --- | --- |
| `vision.plan` | Qwen image features |
| `text_encoder.plan` | Reference-aware language conditioning |
| `vae_encoder.plan` | Image-reference VAE moments |
| `vae_reference_posterior.plan` | Weightless reference sampling/normalization |
| `adaln.plan` | Timestep-dependent conditioning |
| `denoiser.plan` | Original reference-conditioned joint A/V denoising |
| `vae.plan` | Video decoding |
| `audio_vae.plan` | Audio decoding |

`tokenizer.json` and `runtime.json` complete the bundle. The real family
factory recognizes its explicit reference Task header and validates metadata
before loading engines. The legacy text-only bundle path is separate.

## Native request/output contract

The owner exposes `IReferencesTextToAudioVideo` v1.0. Reference images must be
contiguous RGB uint8; audio/video references and initial-frame semantics are
not supported by this implementation. Capacity is checked after original
image preparation; oversized inputs fail rather than being silently resized
to fit the engine profile. Request seed is nonnegative, default 42; reference
posterior sampling separately keeps the original seed 42 per image.

The fixed recipe uses 50 schedule points / 49 denoiser forwards, producing
124 video frames at 24 fps and stereo 32-kHz audio. Default portrait output is
1344 high by 768 wide (4:7, not 9:16). An explicit 1536-by-864 build gives
9:16 geometry, but trained output at that geometry remains unqualified.
The original audio/video endpoint difference is not silently time-stretched.

## Validation boundaries

Family CPU tests and ownership checks:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/minimax_h3/tests --ignore=families/minimax_h3/tests/test_e2e.py \
  -m 'not gpu and not trt' -q
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
```

The owning full trained-video gate is
`tests/test_trained_video_decoder.py`. It compares every pixel of all 124
FP32 output frames against the pinned original reference at unchanged
`rtol=1e-3`, `atol=1e-5`, and checks cold/warm behavior for both presets.
An explicit `TRTMC_MINIMAX_H3_VIDEO_REFERENCE_CACHE` may reuse the exact frozen
full upstream result: case metadata, full input/reference bytes and storage
bound inputs must match their SHA-256 identities. Both new candidate bundles
are built before inference, and all four complete native videos are retained.
This changes execution/storage scheduling, not any frame, comparison dtype or
acceptance threshold. Without that opt-in, the test generates its reference.
Prefix diagnostics, synthetic trajectory tests and guarded factory fixtures
cannot replace that gate or qualify full reference-conditioned generation.
The trained 256x256, one-frame FP32 encoder component comparison also remains
failing: 54 of 12,288 moments exceed its unchanged `rtol=1.3e-6`, `atol=1e-5`.
Diagnostic-only division-order SiLU produced 39 outliers; additionally computing
centered or reparameterized GroupNorm with TensorRT FP32 reductions produced
220 or 187. Each diagnostic variant was built three times and all three outputs
were bit-identical within that variant, but none passed. These candidates were
not installed in the production graph. An affine replay supplied with reference
statistics is an oracle-assisted diagnostic, not a deployable encoder repair.
No component failure or criterion is waived by the executed six-image request.
The complete trained reference encoder, Qwen conditioning, DiT trajectory,
video/audio composition and actual native reference request must all be
verified before advertising end-to-end support.

## Continuation development

Family-private helpers now describe latent-tail continuation using the
[fixed vLLM-Omni reference](https://github.com/vllm-project/vllm-omni/blob/cbe4cc4b7c1126a903553f8e79e44ddfab782cf3/vllm_omni/diffusion/models/minimax_h3/continuation.py).
The guide layout retains static image/text positions and gives the guide and
target the same global media origin. Video guide rows receive the original
0.999 noise augmentation; audio guide rows stay clean at time 1.0. Only target
rows are integrated, and changing clean-audio presence invalidates the AdaLN
cache before the next DiT load. Existing image-only requests retain their RNG
draw order and outputs.

The window planner uses the 17n+5 frame grid, global rounded audio boundaries
and fractional media rotary offsets. Tail extraction and accumulation process
each stereo channel separately and retain the old latent prefix. Two 124-frame
windows with 22-frame hidden overlap have 226 cumulative video frames, 67 video
latent frames and 377 audio latents per channel. Retaining latent prefixes does
not establish an unchanged decoded pixel prefix or a seamless visible boundary.

`reference_continuation_oracle.py` checks packed layout/row times against fixed
upstream CPU code; `reference_continuation_clock_oracle.py` checks exact clock
planning. Native C++ tests use declared engine/RNG doubles for orchestration.
The separate `test_native_reference_continuation.py` requires the real backend,
synthetic BF16 plans and fixed upstream layout source through
`TRTMC_H3_PACKED_SEQUENCE_SOURCE`; it keeps the existing 49-forward trajectory
criterion at `rtol=0.016`, `atol=1e-5` and FP32 samples. A synthetic test does not
qualify trained continuation, decoding, performance or output quality.
`test_native_reference_windows.py` separately compares the complete 98-forward
two-window execution with the fixed upstream model and packed layout, including
the generated parent-to-child guide and cumulative stereo suffix append. It
uses the same FP32 samples and numerical criterion; it also uses synthetic
weights and does not decode or qualify a trained video.

The default explicit builder and pipeline retain the original 124-frame profile.
For development, `reference_build.build(request, writer, continuation=True)`
builds a separate `minimax_h3.image_ref2va_continuation/v1` bundle. Specify
`video_num_frames=226` (or leave it unset); other durations are rejected.
This profile admits zero audio guide rows in window one and 74 in window two.
Its video condition maximum includes seven target-grid latent frames in
addition to the complete static-image budget. At 1344x768 with `six_portrait`,
the maximum is 56,208 condition-video rows and 159,528 total packed rows.
The Qwen visual/text budgets and every reference's preparation remain unchanged.

`ImageReferenceDenoiser::run_windows` serializes the complete two-window request,
uses the original request seed independently for each window, retains all static
references, and appends only the generated latent suffix.
It runs 49 forwards per window (98 total). The runtime decodes all 67 video latent
frames using the original seven-latent static tile engine and builds a new
377-latent audio decoder for 301,600 samples per stereo channel at 32 kHz.
The resulting video is 226 frames at 24 fps. Native video/audio endpoints remain
distinct; neither stream is stretched, truncated or padded to match the other.
Both `minimal` and `original` residency policies apply to the new profile.
An old bundle cannot acquire these operand/decoder shapes by editing metadata.

Only the 226-frame profile declares the optional string runtime configuration
`continuation_prompt`. Empty or identical text reuses the initial conditioning
and preserves the original same-prompt path. Distinct text is encoded separately
with every original static reference. Both windows use the longest complete Qwen
text prefix as their media-time origin, following the fixed upstream per-window
conditioning implementation. Both prompts and operand caps are checked before
the first plan load; malformed child conditioning is rejected before any first
window denoising. A longer child prompt can therefore change the first window's
rotary coordinates as well. This is a program fixed for one bounded request,
not a live prompt update or an assurance that dialogue occurs only once.

The `test_real_native_window_prompts_match_98_forward_upstream` cases retain the
same numerical gate with shorter and longer child text, full 98-forward traces,
clean stereo guides and staged/resident cold/warm comparisons. Synthetic
conditioning parity does not establish trained speech or visible quality.

If a compatible original single-window bundle already exists, the owning
`reference_continuation_build.build(request, writer, source_bundle=...,
source_sha256=...)` entry point rebuilds only `denoiser.plan` and `audio_vae.plan`.
It verifies the complete explicit source digest, the original profile and
checkpoint configuration/statistics, tokenizer, BF16 provenance, and the same
pinned Hugging Face snapshot before loading new weights. It streams the six
unchanged plans through bounded buffers and verifies their content again while
copying. The new bundle records their source and individual digests in
`minimax_h3.reused_plans.json`. Select the same memory preset and reference
capacity as the source bundle. The source compiled plans must be compatible
with the selected backend; this entry point does not perform hardware adaptation.
An incompatible source fails explicitly, and the original bundle is preserved.

This within-request profile does not expose cross-request continuation state,
accept decoded video/audio reference operands,
preserve previously accepted decoded pixels, or qualify trained-model video
quality. Complete trained numerical gates and visible/audio boundary review
remain required before advertising continuation support or realtime generation.
## Experimental OmniRef PDD primitives

`reference_pdd.py` owns the sampling contract and build-time output-head math
for `FastVideo/FastVideo-FastH3-Omni-8-Step-V1`, revision
`53e0d26a1691aa473b6262d8196a8feb2c297df0`. The student partitions a 32-interval
base grid into eight blocks, with separate video/audio shifts. Each block
normalizes and combines its trained output heads in FP32, rounds the fused
parameters to BF16, and retains H3's separate FP32 score and scheduler clocks.
Target noise starts at 0.999; fixed conditioning rows must remain unchanged.

The CPU oracle extracts the original mathematical methods from an explicitly
supplied upstream source checkout:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m \
  families.minimax_h3.tests.reference_pdd_oracle \
  --pdd-source /path/to/FastVideo/fastvideo/layers/pdd.py \
  --scheduler-source /path/to/FastVideo/fastvideo/models/schedulers/scheduling_minimax_h3.py \
  --output pdd-math.json
```

The implementation includes the student graph, 128-token reference-aware VSA
binding and an explicit native sampling recipe. It is not a registered model or
qualified student bundle. Building its exact learned weight partitions and
completing trained-model validation remain required. An eight-step setting on
the original checkpoint is not equivalent.
Neither the CPU checks nor upstream multi-GPU latency establishes single-GPU
performance, output quality or compatibility with existing compiled bundles.

`reference_pdd_builder.py` adds a strongly typed TensorRT output-head selector
and projection component. It stores the eight precomposed banks in BF16,
selects one bank per forward, accumulates the linear operation and bias in
FP32, and explicitly rounds the velocity to BF16 before widening for the
scheduler. The family runtime must validate its internal selector in 0..7.
This component does not implement the student's transformer or attention.

The GPU component oracle compares identical synthetic BF16 banks with PyTorch
at hidden widths 17, 64 and 5376, three dynamic row profiles, and all eight
steps. It retains each plan, input, expected output and actual output. Its
72 enqueues and 144 output comparisons use the existing BF16 numeric limits:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m \
  families.minimax_h3.tests.reference_pdd_projection_oracle \
  --output /path/to/new-output-directory
```

Add `--preflight` to enumerate the cases without creating a CUDA context.
Passing this synthetic component oracle does not qualify the trained student,
128-token VSA, complete audiovisual generation or single-GPU performance.

`reference_vsa.py` now defines the 128-token segment transport, separate
reference-video regions, zero-valid partner tile and TVM-FFI graph boundary.
`kernels/omniref_vsa.cu` implements routing, sparse attention and the learned
compression branch. It includes a generic backend and optional optimized
Blackwell cubins. See `kernels/README.md` for its exact binding contract.

`reference_pdd_dit_builder.py` composes the complete student DiT graph: dense
text refinement, one VSA binding per transformer block, target-only output
selection and the eight fused PDD heads. Original H3 retains its dense path.
The native denoiser selects the eight trained head banks, scales only target
noise by the initial sigma, prepares matching AdaLN times and retains fixed
image/video/audio conditioning. Its explicit bundle schemas are
`minimax_h3.omniref_pdd/v1` and `minimax_h3.omniref_pdd_continuation/v1`.
The latter retains two complete eight-forward windows, the original overlap
and cumulative audio/video decoding geometry. The factory requires the full
released checkpoint/sampling contract and an explicit VSA tile capacity.
Original-H3 schemas retain their original 49-forward recipe. Source presence
does not establish a trained bundle or output quality for either recipe.

The CPU geometry oracle compares the transport with the supplied upstream
functions, including interleaved reference videos and six portrait images:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m \
  families.minimax_h3.tests.reference_vsa_geometry_oracle \
  --base-source /path/to/FastVideo/fastvideo/attention/backends/video_sparse_attn.py \
  --h3-source /path/to/FastVideo/fastvideo/attention/backends/video_sparse_attn_h3.py \
  --output vsa-geometry.json
```

Geometry agreement does not establish GPU attention or full-model agreement.
Pass `--native-probe /path/to/test_minimax_h3_reference_vsa_geometry` to also
compare every native transport index and the FP32 schedule bits. The pinned
contract fixture is copied from the released checkpoint's
`fastvideo_inference.json` at the revision above.

The GPU attention oracle requires the compiled generic TVM-FFI bridge, this
family's kernel DSO and the supplied upstream sources. Run each forced backend
in a separate process because backend selection is cached for that process:

```bash
TRTMC_MINIMAX_H3_VSA_BACKEND=generic \
PYTHONPATH=core/builder:apps/benchmark:. python -m \
  families.minimax_h3.tests.reference_vsa_gpu_oracle \
  --backend generic --output /path/to/new-output-directory \
  --library /path/to/libtrtmc_minimax_h3_omniref_vsa.so \
  --bridge /path/to/libtrtmc_byok_tvm_ffi.so \
  --base-source /path/to/FastVideo/fastvideo/attention/backends/video_sparse_attn.py \
  --h3-source /path/to/FastVideo/fastvideo/attention/backends/video_sparse_attn_h3.py
```

Use `blackwell` for the optimized backend. Each route includes four synthetic
cases; Blackwell also executes one attention operation with the trained target
and six-reference spatial dimensions and head count. Its 4000-row text segment
is synthetic; a complete request also includes Qwen's visual token rows.
The oracle builds dynamic strongly typed
TensorRT component plans, retains all raw operands/outputs, and applies the
existing BF16 `rtol=0.016, atol=1e-5` envelope to every output element. Add
`--preflight` to enumerate all nine cases without initializing CUDA. Component
agreement is still distinct from complete student DiT/AV qualification.

`reference_pdd_build.build` constructs the explicit experimental student bundle
from its exact canonical Hugging Face snapshot. It checks both released
contracts, the widened-head transformer config and the complete 688-tensor
index before loading student weights. Both AdaLN and DiT are rebuilt from the
student. Six unchanged common plans may be copied from an explicitly pinned,
profile-compatible original-H3 bundle after its metadata, provenance and full
content check. The builder admits only base revisions
`9bfb6693f2cf6de171db46d1aa586f67d773a1da` and
`48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc`; their text encoder, tokenizer and
video/audio VAE file identities match. The execution campaign must separately
retain local checkpoint and compiled-artifact verification.

The source bundle must match the requested single/two-window geometry and
memory preset. In particular, a 377-latent audio decoder from a two-window
bundle cannot become a single-window decoder by editing metadata. Student
runtime metadata uses the explicit PDD schemas and records the exact student,
eight forwards per window and all sparse-attention settings. Automatic model
discovery and production App adoption remain gated on complete trained proof.

This build writes the native serialized AdaLN/DiT host buffers directly into
the bundle, then streams the selected common sections in bounded chunks.
It avoids the second plan-sized Python `bytes` allocation and another full
bundle disk copy. The default builder bytes API is preserved. This is a
serialization change, not a precision or sampling change, and does not bound
TensorRT optimizer memory or establish a minimum supported host RAM budget.
