---
title: Benchmarking Reference
---

# Benchmarking reference

The benchmark application sits above TRTMC's public build and task APIs. It
does not register model families or import family implementation code.

## Single-model benchmark

Install the optional benchmark dependencies, then select an installed runtime
root containing the core runtime, TensorRT backend, and required family DSOs:

```bash
python -m pip install -r apps/benchmark/performance/requirements.txt

trtmc-bench list models
trtmc-bench run \
  --model distilgpt2 \
  --runtime-root /opt/trtmc/lib \
  -o results/distilgpt2
```

`trtmc-bench` reads family-owned manifests under
`families/*/tests/manifests/`. Missing bundles are built through the public
builder and cached; pass `--no-build` when every selected bundle must already
exist.

Run a checked-in multi-model specification with:

```bash
trtmc-bench run apps/benchmark/example.yaml -o results/example
```

Use `trtmc-bench --help` and subcommand help as the authoritative option list
for the installed version.

## Timing boundary

Candidate timing measures the public family task call. Bundle construction,
process startup, task loading, warmup, telemetry, and report generation are
outside that measurement. Asset loading is also excluded unless a case
explicitly includes it.

The native worker stops the clock before formatting an observation or
destroying the previous iteration's result. New receipts explicitly include
`observation_serialization_included: false`. Earlier worker revisions included
observation formatting in the measured duration; account for that boundary
change when comparing historical measurements. No comparison threshold changes
with this correction.

The release suite defaults to three warmups and ten measured iterations. A
reference within five percent of candidate p50 is considered equivalent.
Candidate or reference execution failures are operational failures, not slow
performance results.

## Release performance matrix

The matrix coordinates candidate and reference runs without introducing a
second model registry:

```bash
export TRTMC_PERF_WORKER=/opt/trtmc/bin/trtmc_benchmark_worker
export TRTMC_PERF_RUNTIME_ROOT=/opt/trtmc/lib
export TRTMC_PERF_BUNDLE_CACHE=/data/trtmc-bundles

python3 tools/perf_matrix.py check \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml

python3 tools/perf_matrix.py prepare \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate \
  --output artifacts/perf/bundle-preparation.json

python3 tools/perf_matrix.py run \
  apps/benchmark/performance/release.yaml \
  --environment apps/benchmark/performance/environments/gb300.yaml \
  --entry gpt2.generate
```

Preparation is deliberately separate and untimed. Resume or regenerate a
report from stored observations with:

```bash
python3 tools/perf_matrix.py resume artifacts/perf/<run-directory>
python3 tools/perf_matrix.py report artifacts/perf/<run-directory>
```

The release YAML owns model, testcase, operation, measurement, reference, and
comparison semantics. Machine-specific paths belong in the environment YAML.
Reference implementations run in separate processes so their dependencies do
not enter the candidate worker or shared runtime.

## Adding coverage

- Add a weight or profile to the owning family's test manifest catalog.
- Add a matrix entry that names the family, model, operation, workload, and
  reference runner.
- Add benchmark code only when a genuinely new public task interface needs a
  task adapter.

Retain raw observations and reports together with the commit, model revision,
runtime root, target GPU, TensorRT version, warmup count, and iteration count.
Correctness validation should precede performance comparison.

## Semantic Task SDK integration

Migrated bundles select the C Task SDK before execution; non-migrated bundles
retain their existing path. A failed C call is never retried through an old
interface. The worker currently has semantic routes for prompt continuation,
conditional generation, corrupted-text reconstruction, summarization,
image/text generation, point, quantile or joint forecasting, and the following
media, perception and feature workloads. An SDK Task not listed here needs its
own explicit benchmark inputs; a shared operation name does not imply support.

| Operation | Semantic Tasks |
| --- | --- |
| `generate_audio` | `TextToAudio`, `TextToSpeech`, `StreamingTextToSpeech` |
| `transcribe` | `SpeechTranscription`, `SpeechTranslation`, `StreamingSpeechTranscription` |
| `speak` | `SpeechToSpeechResponse` |
| `generate_image` | `TextToImage`, `ImagesTextToImageEdit`, `BatchTextToImage`, `TextToVideo`, `ImageTextActionToVideo` |
| `classify` | `ImageToClassScores` |
| `extract_features` | `ImageToTokenFeatures`, `ImageToPooledFeatures`, `ImageToTokenAndPooledFeatures`, `ImageToSpatialFeatures` |
| `segment` | `ImageToSemanticSegmentation`, `ImagePointsToMasks` center helper |
| `segment_prompted` | `ImagePointsToMasks`, `ImageTextToInstanceMasks` |
| `disparity` | `StereoImagesToDisparity` |
| `encode` | `TextToPooledFeatures`, `TextToTokenFeatures` |
| `embed` | `TextToEmbedding` |
| `rerank` | `TextQueryDocumentsToRelevance` |
| `control` | `ImageStateToActionChunk` |

For semantic Task manifests, explicitly specified generation controls retain
their types; omitted controls use the family defaults. A testcase can supply
additional family-declared keys through its `config` object. Flat and nested
duplicate keys are errors, not an implicit override mechanism. Model data stays
in typed request fields, never in Config. A quantile or channel axis is not a
request batch. Joint forecasts retain both point and quantile outputs from one
family evaluation.

Audio SDK inputs retain WAV sample rate and interleaved channels. Duration uses
frames divided by sample rate; `num_samples`/`output_samples` remain scalar PCM
counts, with explicit `channels` and `output_frames`. Transcription retains text,
token IDs and the family's actual segment timestamps. The worker's existing
`max_new_tokens` spelling maps to the declared `max_output_tokens` Config key for
one-shot transcription/translation only; streaming ASR keeps `max_new_tokens`.
`language` is the typed ASR source language or TTS language, and translation has
a separate optional `target_language`. Absent languages use family defaults.

Streaming ASR measures a fresh stream for every invocation, including creation,
chunk submission, finalization and release. Its separate `first_partial_ms`
clock begins after stream creation. Packetization defaults to 160 ms and never
splits an interleaved frame. Streaming TTS measures the direct synchronous
callback API and counts delivered PCM without file writes or an extra worker.
An explicit `streaming` input must agree with the semantic Task. Failed or stopped
calls are operational failures, never completed measurements. Native-batch
audio and dialogue-session benchmarks are not included in these routes.

Image batches make one native batch call with ordered prompts and optional
`seeds`/`item_configs` arrays. Their lengths must match the request count, and
shared/item duplicate keys are rejected. Scalar replay uses
`initial_latents_path`; this does not define a broadcast input for a batch.
Video metrics count actual clips and frames with their returned timestamps, not
an inferred FPS. A worker-only completion has no produced media and fails this
single-process benchmark instead of counting as an image or video.

Pooled and token features remain different workloads. The joint image feature
Task returns both hidden and pooled arrays from one call. Classification keeps
the original score representation, and document-list reranking preserves order;
the list interface alone is not evidence of native GPU batching. SAM's `segment`
helper uses a center foreground point and thresholds the **first** family-selected
mask at zero; explicit points use `segment_prompted`, which keeps all masks.
Empty masks remain empty. Confidence and predicted IoU are reported separately.

Disparity writes the final F32 map beside the result JSON after measurement,
retaining dimensions and the left-grid `x_left_minus_x_right` convention. File
flush errors fail the run. New semantic image/feature routes honor an explicit
`asset_loading_included=true`; older legacy image/feature paths did not all honor
that flag, so historical comparisons must account for this boundary. Stateless
action chunks preserve schema, values and timing; they do not emulate a queue.

`trtmc_dataset_benchmark` preserves its separate, explicit benchmark workload:
12000 maximum new tokens, temperature/top-p 1, top-k 1, min-p 0, seed -1,
chat formatting off, thinking on, boxed-answer stopping off and check interval
16. These defaults apply only to controls declared by the selected Task; an
explicit unsupported flag fails even when its value equals a default. They are
application workload choices, not shared runtime or QuickStart defaults.
`--set NAME=VALUE` accepts additional declared scalar/list controls on semantic
bundles, using the same text parser as the CLI. Each successful sample records
the exact `submitted_config`; absent, family-computed values are not fabricated.

For nonnegative base seeds, both dataset paths add `seed_index` when present,
otherwise the sample's zero-based row index. Overflow fails instead of wrapping.
Dataset timing ends before answer extraction and JSON output. Preserve the
bundle, submitted parameters and software revision when comparing these runs.
