# Clef structured decisions

This family implements the released Clef and Clef-Flash checkpoints:

| Checkpoint | Revision |
| --- | --- |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | `2f3de3dd85f379784083b0814d997ab627200f0c` |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | `fde727a287004204b7518dcc983fe64379776712` |

Each bundle uses its checkpoint's complete backbone, vision encoder, output
embeddings, and joint schema head. The family evaluates
the complete schema in one forward pass and returns per-option probabilities;
there is no autoregressive answer generation.

The reference is the release's `joint_schema_model.py` with Transformers 5.10.2
and BF16 weights. The production runtime is C++ and executes native TensorRT
graphs through ModelConnect. Python and Torch are used for building and
reference validation, not inference.

## Build and run

Start with the repository's [source-build environment](../../website/docs/getting-started/source-build.md).
Build the native CLI, TensorRT backend, and Clef runtime:

```bash
cmake --build build --parallel 8 --target \
  trtmc trtmc_backend_trt trtmc_model_clef trtmc_cli_clef

build/trtmc clef build Cloudflare/clef \
  --revision 2f3de3dd85f379784083b0814d997ab627200f0c \
  --max-sequence-length 512 -o clef.bundle

build/trtmc clef decide clef.bundle \
  --runtime-root build --record families/clef/tests/fixtures/invoice.json

build/trtmc clef decide clef.bundle \
  --runtime-root build --record families/clef/tests/fixtures/outage.json

build/trtmc clef build Cloudflare/clef-flash \
  --revision fde727a287004204b7518dcc983fe64379776712 \
  --max-sequence-length 512 -o clef-flash.bundle

build/trtmc clef decide clef-flash.bundle \
  --runtime-root build --record families/clef/tests/fixtures/clef-flash-outage.json
```

The fixtures reproduce the two complete text examples on the model card. The
invoice record produces `overdue`; the outage record selects `technical`.
The response contains `model`, `answers`, and `usage`, including zero generated
output tokens. The direct-record form can omit `model`; its response uses
`"clef"`. A supplied model string is returned unchanged.

The builder uses BF16, one GPU, and an explicit input capacity. The example
above uses 512 tokens. `--max-questions` and `--max-options` control schema
capacity. As in the reference encoder, the complete schema is retained and the
state is truncated to the remaining capacity. An oversized schema is rejected.
`--max-state-tokens` can impose an additional state limit at inference time.

## Images and video

The CLI accepts image filenames in `images` and ordered arrays of frame
filenames in `videos`. Paths are relative to the record file. For example:

```json
{
  "state": {"task": "Review the attached receipt."},
  "images": ["receipt.jpg"],
  "questions": {
    "legible": {"type": "noul", "instructions": "Is the receipt total legible?"}
  }
}
```

The card does not distribute `receipt.jpg`. `tests/media_fixtures.py` creates
a deterministic receipt and video for paired native/reference checks; those
assets are test inputs, not upstream images. Native preprocessing preserves
frame sampling, patch ordering, timestamps, and multimodal position IDs.

For direct C++ integration, load the bundle with `trtmc::load_task`, obtain
`trtmc::IStructuredDecision`, and call `decide(StructuredDecisionRequest)`.
The request holds the JSON record and optional RGB image/video arrays with
pixel values in `[0,255]`. The result includes the formatted response and
per-question option IDs, logits, and probabilities. Calls may use different
records and media while reusing the loaded model.

## Validation and measurement

Enable `TRTMC_BUILD_TESTS` to build the family probes. The family target depends
on them in test builds, including the CI build path.

```bash
TRTMC_E2E=1 TRTMC_RUNTIME_ROOT="$PWD/build" \
TRTMC_NATIVE_BUILD_DIR="$PWD/build" \
TRTMC_CLEF_CHECKPOINT=/path/to/pinned/clef \
TRTMC_CLEF_BUNDLE="$PWD/clef.bundle" \
PYTHONPATH=core/builder:apps/benchmark:. \
python -m pytest families/clef/tests/test_e2e.py --e2e-model clef -q
```

The E2E gate compares every option logit and probability against the original
implementation, requires matching selected options and repeated outputs, and
checks that the native process does not load Python, Torch, or c10. Separate
tests compare token IDs, spans, media patches, and recurrent attention.

Select `--e2e-model clef-flash` for Flash only. Local Flash artifacts can be
provided through `TRTMC_CLEF_FLASH_CHECKPOINT` and `TRTMC_CLEF_FLASH_BUNDLE`.
`--e2e-model clef` selects both checkpoints in this family.

The standard benchmark operation is `decide`:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python -m trtmc_benchmark.cli run \
  --manifest-root families --model clef --case clef-invoice \
  --bundle clef.bundle --no-build --runtime-root build \
  --worker build/trtmc_benchmark_worker --warmup 5 --iterations 30 \
  --output clef-benchmark
```

The measurement boundary is the complete public Task call, including record
encoding and response construction. Model loading, compilation, warmup, and
image-file decoding are excluded. `tests/benchmark_compile.py` measures the
same boundary with `torch.compile(mode="max-autotune")` and checks compiled
accuracy before timing.
Pass `--model clef-flash` to select Flash's pinned checkpoint metadata and fixtures.

On GB300 with TensorRT 11.1.0.106, all seven E2E cases pass against the pinned
original implementation. This includes the two model-card text examples,
alternative answers, multilingual input, a receipt image, and video frames.
The largest absolute option-probability difference is 0.002188; the BF16 gate
uses `atol=0.002, rtol=0.01` for probabilities and `atol=0.125, rtol=0.015` for
logits. A 16,384-token request also passes those gates. Reusing one native Task
across text, image, video, and maximum-length requests preserves repeatability.

Measured warm median Task latency is 41.5 ms for invoice, 42.3 ms for outage,
42.7 ms for receipt, and 45.4 ms for video (five warmups and 30 samples each).
The compiled text reference takes 55.4 ms and 64.4 ms respectively. Both use
the full request boundary described above. These numbers exclude the native
bundle's approximately 186-second load time.

The Torch 2.12.0+cu130 media baseline initially fails in an Inductor
masked-scatter scan kernel on this setup. `benchmark_compile.py` provides the
explicit `--aten-masked-scatter` option to compile the remaining model while
retaining the original ATen operation; each receipt records this fallback.
It also regenerates graph IR because cached graphs do not reflect a changed
decomposition table. Keep compiler failures separate from completed timing
samples when comparing the implementations.

Stricter intermediate vision-feature and synthetic layer diagnostics still
have numerical mismatches. Their thresholds have not been relaxed. These
component results remain distinct from the passing end-to-end output gates;
this is not a general hardware or release-readiness claim.
