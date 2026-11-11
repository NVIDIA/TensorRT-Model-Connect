# D3 dense native vertical slices

**Current qualification:** The selected Edge GPU E2E fails against the pinned
original reference at the existing tolerances. None of the five profiles is
qualified for serving. Results below are scoped to the specific historical
bundles and experiments stated; native batch consistency is not reference parity.

The first native vertical slice was validated with `vllm-sr/d3-edge` revision
`bee5e872f75b9c5d1405767bf244a6948b95c0af` on an RTX 5090, driver 580.173.02,
CUDA 12.9 and TensorRT 11.1.0.106. The original reference used PyTorch
2.14.0+cu130 and Transformers 5.17.0, with the unmodified checkpoint runtime.
An earlier single-row bundle passed its then-current E2E for fixed text and RGB8 256x256 image records,
covering all three question types and text/image/text reuse of one Task.
This is evidence for that bounded case, not broader quality, performance,
hardware or model-family qualification.

A supplemental image-only record (`state="Use the attached image."`, the same
questions and red RGB8 image) passed the probability gates but failed the raw
logit gates for `choice` and `noul` against the default original reference.
The maximum absolute probability difference was 0.003137; combined absolute
and relative gates still passed. A separate diagnostic with PyTorch's
`allow_bf16_reduced_precision_reduction=False` passed both gates on that record.
This diagnostic did not replace the production reference or acceptance rules.
Default-reference raw-logit parity beyond the declared fixture remains an
open numerical qualification boundary.

The preceding version-2 batch bundle passed B1/B2/B4/B8, overflow chunking, mixed lengths
and images, permutation, empty input and invalid-input recovery against
independent native calls. The expanded original-reference comparison passed
8 of 9 records. The blue-image brightness record failed probability parity
(maximum absolute difference `0.007269` on that bundle); the same
record also fails in the preceding single-row runtime. The full selected E2E
therefore currently fails. The first observed divergent visual boundary is
patch projection; the accumulation-mode diagnostic does not resolve this case.
Those earlier timed text/red-image outputs passed their unchanged gates.

The subsequent numerical investigation restores BF16 normalized Q/K storage,
FP32 sigmoid/elementwise opmath with BF16 result storage, and reciprocal-power
RoPE frequencies. Visual Linear now accumulates in FP32, adds bias, and stores
one BF16 result rather than introducing a BF16 dot-output boundary before bias.
On identical inputs, FP32 sigmoid followed by BF16 storage matches the original;
reciprocal-power frequencies also match its BF16 RoPE cosines/sines.

These corrections do not establish end-to-end parity. The earlier B8/S2048 build
passed the original text/red-image/text sequence and native batch-versus-single
checks, including an equal-length red/blue pair. It failed the expanded
original-reference gate: blue-image brightness has maximum probability error
`0.010874` (`0.231399` versus `0.220525` for the first option). The additional
equal-length red control also fails its original-reference comparison. Across
all 14 independent reference comparisons, 11 pass and 3 fail; two failures are
the same blue request included in both the mixed and equal-length probes.
The selected E2E for that build failed. Current performance was not remeasured.

An independent original-only control repeats the identical blue prompt at
B1/B2/B4/B8. Repeated B1 is exact, but B2 and B4 differ from B1 by `0.007724`
and `0.011214` respectively and fail the same probability gate. Disabling
PyTorch BF16 reduced-precision reductions changes B1 by `0.007367` and also
fails that gate. Replacing only the original BF16 Linear accumulation with an
FP32 diagnostic changes probabilities by `0.007032` and fails. These are
diagnostic controls, not replacement references or revised acceptance rules.
On identical first-layer projection inputs, TensorRT and the original BF16 a/b
projections already differ by up to `0.03125`; the TensorRT outputs are closer
to a separate FP32-accumulation control. Both the visual path and text stack
contribute to the final mismatch. The precise remaining layer/rounding
correction has not been established. Batch equivalence does not establish full
original-model parity. Tolerances remain unchanged, and this slice is not
qualified for Triton integration.

The B8 / 2048-token bundle reserved approximately 19.0 GiB of process GPU memory
on the validation host even at B1. Current backend contexts allocate for the
maximum profile; lower observed batch occupancy does not release that reserve.
Account for the full profile when planning serving instances.

## Dense family adaptation

The family graph and runtime use checkpoint hidden widths, layer counts,
attention heads, convolution widths and recurrent-state dimensions. Mini and
flash repeat normalized Q/K heads to their 32 value heads, while edge/lite/nano
use 16 key and value heads. The response model identity follows the matched
profile. Version-3 bundles prevent older edge-only runtimes from loading a
larger model with an incorrect layer count. The new runtime also accepts
legacy version-2 edge bundles. All five profiles retain BF16 backbone storage and FP32 recurrent
state/readout arithmetic; no quantization format is implemented.

| Profile | Hidden width | Text layers | DeltaNet key/value heads | Vision layers |
| --- | ---: | ---: | ---: | ---: |
| edge | 1024 | 12 | 16/16 | 12 |
| lite | 1024 | 24 | 16/16 | 12 |
| nano | 2048 | 24 | 16/16 | 24 |
| mini | 2560 | 32 | 16/32 | 24 |
| flash | 4096 | 32 | 16/32 | 27 |

The 2026-11-11 isolated RTX 5090 qualification built real B2/S512 bundles with
TensorRT 11.1.0.106 and executed text, images and native batches for every row
below. CLI output matched the reused Task. The original oracle used PyTorch
2.14.0+cu130, Transformers 5.17.0 and FLA 0.5.2 with the Torch convolution
fallback and unchanged BF16 reduction defaults.

| Profile | Build and native batch equivalence | Original comparisons passed |
| --- | --- | ---: |
| edge | Passed | 10/14 |
| lite | Passed | 7/14 |
| nano | Passed | 10/14 |
| mini | Passed | 12/14 |
| flash | Passed | 11/14 |

The 14 comparisons include the original text/image/text sequence, nine varied
independent requests, and the equal-length red/blue pair. Repeated requests are
intentional state-isolation controls; these counts are not a quality score.
A separate fresh-build pytest E2E for mini also failed a confidence comparison
(`0.372998` versus `0.363849`, maximum difference `0.009149`, unchanged
`atol=0.004, rtol=0.01`). Its artifact differs from the first qualification
bundle; keep both results rather than treating a single build as a guarantee.
Every profile still has numerical failures under the original gates. Native
execution and batch equivalence must not be advertised as full correctness or
serving readiness. Quantization, Triton integration and performance qualification
remain outside this adaptation. No services were restarted or republished.

`tests/manifests/` pins one upstream revision per profile. Configuration fixtures
record their provenance under `tests/fixtures/profiles/provenance.json`.
Recognition, graph implementation, native compilation and successful numerical
qualification are separate evidence levels. In particular, the unresolved edge
parity failure above is not removed by extending the family.

Select an individual GPU gate with `--e2e-testcase d3-lite-text-image` (or the
corresponding profile name); `--e2e-model d3` selects all five. New profiles use
B2/S512 test bundles to bound qualification resources. The maximum supported
build capacity remains eight question rows, subject to host GPU memory. Build
and validate each target profile on its intended hardware before serving it.

## Ownership and execution

`support.py` selects `structured_decision` with BF16 by default. Its explicit
checkpoint marker takes precedence over a generic architecture match. The builder recognizes the released `edge`, `lite`, `nano`, `mini`, and `flash`
dense geometries. `d3`, `turbo`, `ultra`, alternate geometries and quantized
checkpoints remain outside this adapter. D3 owns its weight mapping, TensorRT graphs, tokenizer,
request formatting, image preparation, orchestration and correctness oracle.
The tokenizer, visual topology and DeltaNet graph were derived from this
repository's Qwen3.5 family and are intentionally independent copies.
The pre-tokenizer freezes Unicode 15.0 letter, mark and number categories;
characters introduced in a later Unicode version have not been qualified.

A bundle contains embedding, visual, per-layer and readout plans, a tokenizer
and runtime configuration. Each full-attention layer consumes the complete
prompt with bidirectional attention. Each DeltaNet layer consumes that layer's
input tokens in order, starting from zero convolution and FP32 recurrent state.
D3 linearizes its patch embedding to `F.linear`; visual projections use BF16
inputs/outputs with explicit FP32 accumulation and bias before output storage.
The last token passes through the final norm and separate FP32 255-code
readout. Temperature and softmax apply only to the question's option codes.
There is no autoregressive decoding, custom GPU kernel or hardware dispatch.

The version-3 runtime packs independent question rows into true TensorRT
batches. DeltaNet advances every row in a batch together, retaining separate
FP32 recurrent and BF16 convolution states on the GPU. Full attention uses
right-padding key masks; readout selects each row's last real token. Images
have a separate batch axis in visual attention. No attention or recurrent
state crosses request boundaries.

## Initial supported inputs

- Text or JSON state; questions in insertion order with `choice`, `noul`, `score`.
- One optional decoded RGB8 image, shared by every question. Pixels reach the
  Task as RGB HWC floats in `[0,255]`, representing integral RGB8 values.
- The image must exactly match the bundle image profile, default `256x256`.
  There is no resize or implicit input truncation.
- Build capacity `max_batch_size` from 1 to 8 question rows; BF16 backbone,
  FP32 recurrent state/readout/probability arithmetic, TP=CP=1.
- `IStructuredDecision::decide` remains available. The optional
  `IBatchStructuredDecision::decide_batch` accepts a vector of independent
  requests and returns one result per request in the original order. Questions
  are flattened and processed in capacity-sized chunks, including calls larger
  than the engine batch limit. Empty input returns an empty vector.
- The complete batch is validated before inference. An invalid request fails
  the whole call. Different text lengths and mixed text/image requests are
  supported; token usage excludes padding. Every call starts with fresh state.
- Literal chat and media control tokens in request text are rejected before
  tokenization. Checkpoints and their templates must come from trusted sources.

Video, multiple images, alternate model geometries, prefix reuse, permutation
averaging, Box/Track, quantization and generation are not implemented.
Unsupported input/options throw instead of silently selecting another path.
Unlike the original System One endpoint, this initial native API fails the
whole call on malformed/over-capacity questions, rather than returning
per-question error documents. It is not an HTTP or jev-sdk adapter.

## Prepare and build on a compatible host

Follow `website/docs/getting-started/source-build.md` and
`apps/devtoolkit/README.md`. Use a checkout-owned environment prepared with
`family="d3"` and install only `families/d3/requirements.txt` in addition to
the documented build prerequisites. Use a disk with room for the development
image/toolchain, checkpoint, staging sections, bundle and compiler workspace.
The reference interpreter must be separate from the builder's pinned
Transformers environment.

Inside the prepared builder environment, from the checkout root:

```bash
export PYTHONPATH="$PWD/core/builder:$PWD/apps/benchmark:$PWD"
python -m tensorrt_model_connect build vllm-sr/d3-edge \
  --revision bee5e872f75b9c5d1405767bf244a6948b95c0af \
  --precision bf16 --max-sequence-length 2048 --max-batch-size 8 \
  --image-height 256 --image-width 256 \
  --output /path/on/artifact-disk/d3-edge.bundle
```

The image profile can be another multiple of 32 within the checkpoint's pixel
limits. Inputs that would exceed the sequence capacity are rejected.

Configure the native runtime using the documented source-build CMake options,
with `TRTMC_BUILD_TESTS=ON` to include the family probe. Set the CUDA architecture
from the actual GPU as documented; do not copy a value from another host.
Build only the required targets:

```bash
cmake --build "$TRTMC_BUILD_DIR" --parallel 2 --target \
  trtmc trtmc_backend_trt trtmc_model_d3 trtmc_cli_d3 d3_task_probe d3_batch_task_probe
"$TRTMC_BUILD_DIR/trtmc" d3 decide /path/on/artifact-disk/d3-edge.bundle \
  --record families/d3/tests/fixtures/record.json \
  --runtime-root "$PWD/$TRTMC_BUILD_DIR"
```

For an image, append `--image-rgb /path/to/image.rgb`. The file is raw RGB8
HWC bytes without a header, exactly matching the bundle image dimensions.
This codec-free CLI input exercises the same decoded tensor Task as C++ callers.

## Validation

Source-only tests compile and execute native host helpers when a C++ compiler
and nlohmann-json headers are available. The optional `nlohmann_json` Python
package supplies headers if the system does not. No TensorRT or GPU is required
for these tests. Frozen prompt, token and answer fixtures were generated from
the original checkpoint's `d3_format.py`, `d3_runtime.py` and chat template,
without loading its weights. The fixture records the original source revision.
`image-reference.npz` additionally records exact BF16 pixel patches and FP32
position interpolation inputs from Transformers 5.17.0; its JSON sidecar
records the deterministic image, processor and dependency versions.

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest families/d3/tests -q
PYTHONPATH=core/builder:apps/benchmark:. python -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python tools/test_impact.py --validate
git diff --check
```

Set `TRTMC_D3_TOKENIZER` to the same revision's `tokenizer.json` to additionally
compare native token IDs with the frozen reference IDs.

The GPU test always builds a real bundle from the manifest's exact checkpoint,
checks automatic family resolution, and compares the native CLI and a reused
Task against the original model for text/image/text, including all three
question types. It checks logits, option probabilities, decisions, usage,
normalization and repeatability. The initial parity gates are absolute/relative
`0.05/0.01` for logits and `0.002/0.01` for probabilities. These combined absolute/relative gates were fixed before GPU execution and
were not relaxed during validation. Measured maximum absolute probability
differences were 0.001325 for text and 0.002943 for the image; the latter passes
the combined gate, not a standalone 0.002 absolute bound.

Prepare a separate GPU reference interpreter with `transformers==5.17.0`,
`torch>=2.8`, matching torchvision, Pillow, accelerate, safetensors and
huggingface_hub. The checkpoint's own `requirements.txt` describes its tested
reference environment. Flash-linear-attention and causal-conv1d are optional;
record their presence and versions because they affect reference numerics.
Then run inside the prepared builder environment:

```bash
export TRTMC_BINARY="$PWD/$TRTMC_BUILD_DIR/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/$TRTMC_BUILD_DIR"
export TRTMC_D3_REFERENCE_PYTHON=/absolute/path/to/reference-env/bin/python
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/d3/tests --e2e-model d3 -q
```

Selected GPU tests fail if prerequisites are missing; they do not silently
substitute a stub or a different reference. The batch probe additionally checks B1/B2/B4/B8, overflow chunking,
permutation, mixed lengths, an equal-length red/blue pair, invalid-input
recovery and empty batches against independently executed requests. The E2E
compares those independent requests with the original checkpoint using the
same gates. These fixtures do not establish broader model quality. Keep experiment outputs in `benchmark-jev`; measure
native inference and HTTP overhead separately in the later comparison stage.

## Third-party provenance

Prompt/answer semantics and frozen host-reference outputs derive from the
Apache-2.0 D3 release:
https://huggingface.co/vllm-sr/d3-edge/tree/bee5e872f75b9c5d1405767bf244a6948b95c0af.
The native port is modified for Model Connect's Task and bundle contracts.
The upstream NOTICE is preserved here:

```text
d3-edge (Decision 3.0)
Copyright 2026 vLLM Semantic Router Team

Built on Qwen/Qwen3.5-0.8B (Apache-2.0).
```

Qwen-derived repository code retains its NVIDIA Apache-2.0 headers. No model
weights are included. Processor reference fixtures were produced with the
Apache-2.0 Transformers package; see the image fixture's provenance sidecar.
