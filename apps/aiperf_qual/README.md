# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).

- **Acc**: TRTMC must be as accurate as the native model. Where a gold-labelled benchmark fits the
  model (`absolute`, most models), both sides answer it and are scored against the gold answers, and
  the two scores must be close. Tasks without one (generated media, robot control, stereo, depth,
  image features, speech-to-speech) are judged by the family's own benchmark qualification cases;
  random-weight test models are Perf only (`accuracy_source: none`).
- **Perf**: TRTMC must be faster than the native model (eager and, where listed, `torch.compile`)
  at the candidate's precision, timed over the whole Task call on both sides.

## Design

| Layer | What | Model-specific? |
|---|---|---|
| Execution | Two HTTP servers speaking the same `/v1/tasks/{operation}` protocol: TRTMC (`trtmc-perf-serve --backend trtmc`, later `trtmc-server`) and the native reference (`--backend reference` generic adapters, or `--backend script` for the family's declared reference). AIPerf sends every request and computes the statistics. | No |
| Family cases | `families/<family>/tests/benchmark/*.yaml`: the family's accuracy cases, its candidate build, and its performance workload request. | Family-owned |
| Task | `config/tasks.yaml`: per catalog Task, its gold-labelled benchmarks (`absolute`), checks for every model of the Task, and Perf settings. | No |
| Model | Derived from the catalog and the family cases. `config/models/<profile>.yaml` holds only exceptions. | Only exceptions |

`trtmc-aiperf-qual plan` prints the derived configuration of every model.

### Accuracy

**Absolute accuracy** (`absolute` in `config/tasks.yaml` and `config/models`; `absolute.py`). TRTMC
and the native model (the generic adapter, eager, at the candidate precision) answer the same
problems and each side is scored against the gold answers. An entry passes when the two scores
differ by at most `max_delta_points` (or `max_relative` of the native score): "close" is a size; the
paired McNemar test (right/wrong metrics) or bootstrap interval (corpus metrics) is a note. A model
whose catalog request samples answers once per seed on each side (mean scores compared). Benchmarks
whose prompts need a longer bundle than the catalog's (`sequence_length`) build `<profile>-qual-<length>`;
problems that still do not fit are dropped for both sides. Every entry also reports the model-call
time of both sides on the benchmark's own requests (informational). With `native_replicas` in the
environment, the native side runs as that many copies of the adapter (as many as fit the GPU), each
answering one problem at a time: the answers are the same, the native model-call times are then not
comparable (the workload light stays white). When only the family's script reference serves the native model (one process
per request for most), each benchmark takes at most 300 problems on both sides (MMLU: 5 per subject).

| Benchmarks | How | Models (examples) |
|---|---|---|
| MMLU 5-shot (57 subjects x 20); GSM8K (first 500) and MATH-500 are defined but off by default (their native eager generation dominated the run time) | AIPerf's own benchmarks and graders at pinned dataset revisions (`trtmc_aiperf_plugins.benchmarks`), completions or chat by the catalog request | text LLMs |
| LAMBADA, TinyStories (last word), BART span reconstruction | AIPerf accuracy plugins with first-word / sentence graders | small LMs, TinyStories, BART |
| MMStar, OCRBench, RefCOCO, LibriSpeech WER, STS-B Spearman, SciFact retrieval / rerank nDCG, HumanEval pass@1, ImageNetV2 top-1, COCO mAP, ADE20K mIoU, point / text-prompted mask IoU, ETTh1 MSE, WMT / FLORES chrF++ | gold suites sent through AIPerf's `trtmc_task` endpoint, scored by `gold_metrics` | VLMs, ASR, encoders, rerankers, code models, classifiers, detectors, segmenters, forecasters, translators |
| GenEval-style (images), VBench object dimensions (videos), MagicBrush CLIP-I (edits), SeedTTS ASR round trip | supplementary checks rendering both sides (`geneval`, `edit_similarity`, `tts_intelligibility`) | diffusion, edit, and TTS models |

**Family cases** (models whose Task has no gold-labelled benchmark). Each accuracy case of the family
runs its own evaluation (`qualification_tests.benchmark_qualification.accuracy.run_accuracy`: dataset,
selection, native reference, metric, gate) unchanged; only the TRTMC side is replaced: its requests
go through AIPerf to a persistent trtmc-perf-serve session. A failing case is re-run on an isolated
session (a fresh worker per request): passing there means state leaks across requests
(`acc-session-state`). A failure the native model shares at the candidate precision, or of a case
that samples, is `acc-inconclusive`. The candidate bundle follows the family's candidate build (a
`<profile>-qual` bundle when it differs from the catalog manifest). A model with neither benchmarks
nor family cases is a configuration error unless it declares `accuracy_source: none` and an
`accuracy_note` (Perf only, for example random weights).

**Checks for every model of a Task** (`supplementary`):

- Text-to-speech: an ASR round trip (`tts_intelligibility`): TRTMC and the native model speak 200
  SeedTTS sentences, Whisper large-v3-turbo transcribes both, and TRTMC's mean word error rate
  against the text must stay within 5 points of the native model's. Sampling models (Bark,
  MagpieTTS) cannot be judged per sample, but what they say can.
- Text-to-image (`geneval`): GenEval's requirements (objects, counts, colors, positions) on 100 of
  its prompts over its six tags, checked on both sides with OWLv2 detections and CLIP colors;
  TRTMC's pass rate within 2 points of the native model's. Videos: VBench's object dimensions (20
  prompts, the middle frame, at most one prompt apart).
- Image edits (`edit_similarity`): 100 MagicBrush edits against the human targets (CLIP-I), TRTMC's
  mean within 1 point of the native model's.

Generated media drift between TRTMC and the native model at any precision, so for image generation
and edits these gold-referenced checks decide Acc, and the pixel comparisons are reported only
(`informational`, never in the verdict): the family's media parity cases
(`family_cases_informational`), and the pixel parity under latent replay (`replay-parity`).
Families whose TRTMC runtime accepts caller initial latents (Flux, PixArt, Qwen-Image, Z-Image, Wan
2.1) render with the same initial noise on both sides (latent replay, `trtmc_perf_serving.latents`;
GenEval and MagicBrush use it too), so the first 10 prompts (3 videos) are compared pixel by pixel:
how much further TRTMC is from a native fp32 render than the native half-precision run is (PSNR /
SSIM), or each sample against the native output without a usable fp32 render. `recheck` runs these
checks again on finished results (reusing generations that sent the same requests); `rejudge`
reports results the configuration no longer asks for as informational.

A suite with `base: catalog` overrides the profile's catalog request (size, steps, seed, ...) with
its dataset fields. Exceptions stay in `config/models/<profile>.yaml`: the benchmarks (`absolute`,
with overrides such as `{name: mmlu, per_task: 10}`), `absolute_sequence_length` (a cap for models
whose longer bundles cannot run), quantization (`candidate.quantization`, which selects the
benchmarks' `quantized_gate`), `reference.timing_precision` (time the native model at another
precision when it is invalid at the candidate's), reference options, and build overrides.

Generated media leave the servers as digests (`trtmc_perf_serving.digests`): 64x64 thumbnails of
the first/middle/last frame and a banded log spectrum for audio, which the Perf output check compares.

### Performance

**L1** sends one request repeatedly to each server (warmup, then N requests, R runs): the family's
performance workload request when it declares one, else the catalog testcase. The metric is the
server-side time of the whole Task call on both sides (`trtmc_model_call_time`: TRTMC
`public_task_call_wall`; references including input preparation and output decoding, with the
model-only time kept as `model_only_ms`). Per side, the per-run p50 values give a mean and a 95%
Student-t CI; `torch.compile` is credited with its best run, and compiled causal-LM references use a
static KV cache when the architecture supports one. Lights follow the perf matrix rule: green faster
by more than 5%, red slower by more than 5%. A comparison is white when outputs differ, a mean side's
CI exceeds 5% (and 0.05 ms), the native model had to be timed at another precision than the
candidate's (it failed there), or the GPU was busy (>= 20% utilization) with other processes right
before a timed run. For generated media the output check compares geometry only. Family script
references built on the shared harness are loaded once and timed per request like the adapters;
other scripts time themselves in their own process (noted in the report). Timing phases hold the
host GPU lock (`gpu_lock`), which bundle builds on the same host also take.

**L2** (informational; it never changes the category):

- Text generation: an AIPerf sweep over `/v1/completions` with synthetic prompts of a fixed length at
  each concurrency level, for TRTMC and the native model; the report compares request throughput and
  latency percentiles. trtmc-perf-serve serializes requests, so higher concurrency measures queueing
  rather than batching.
- Image and video generation: AIPerf's `image_generation` / `video_generation` endpoints (the OpenAI
  Images and Videos APIs, polled) send three PartiPrompts at the catalog request with the denoising
  steps at half and at the catalog count. Each server runs alone on the GPU with `--memory-probe`
  (NVML peak device memory during each call, minus the use before the model loaded); the server
  records give the model-call p50 and peak memory, and the two step counts split the call into a
  per-step (denoiser) and a fixed part (text encoders, VAE decode). The light compares the
  model-call p50 at the catalog steps. The servers also return SGLang's `inference_time_s` and
  `peak_memory_mb`, which AIPerf reports as video inference time and peak memory.

### Reference environments

Families declare reference requirements in their benchmark qualification case
(`families/<family>/tests/benchmark/*.yaml`, `reference_environment`). The reference server runs in
that family's environment, created and cached by `trtmc-perf-serve reference-env` (same digest
rules as benchmark qualification, layered on the base interpreter). Bundles are built in the same
environment, which carries the family's declared requirements as CI installs them. The orchestrator
and AIPerf run in their own environment; the TRTMC server needs no Python model packages.

### Reference backends

`reference` (generic adapters, persistent, eager/compile) serves every operation it supports; if it
cannot load or run a model, the family's declared qualification reference (`script`) takes over.
World-model generation and text-prompted segmentation always use the family reference, because
the generic adapters would ignore their extra inputs; image edits run on the Diffusers adapter (input
images, Qwen-Image's true CFG scale) and fall back to the family reference. A `script` reference built
on the shared harness is loaded once per server; other scripts start one process per request and
report their own warmup/iteration p50.

The Diffusers adapter ties a T5-style text encoder's `encoder.embed_tokens` back to `shared` when
loading left it zero: Transformers 5.2 does not tie it for `UMT5EncoderModel` (Wan), every prompt
then encodes to zeros, and the pipeline renders the same output for any prompt (the server log notes
the repair). A perf request that leaves `num_steps`, guidance, CFG, or frames at -1 takes the value
the family request or the catalog states, since TRTMC and Diffusers apply different defaults.

## Setup

```bash
apps/aiperf_qual/setup.sh /path/to/aiperf-venv /path/to/trtmc-venv/bin/python   # orchestrator + serving packages
cp config/environments/example.yaml config/environments/<machine>.yaml         # fill in this machine's paths
trtmc-aiperf-qual doctor --environment config/environments/<machine>.yaml       # paths, packages, GPU, model list
```

## Commands

```bash
E=config/environments/gb300-perf-serving.yaml
trtmc-aiperf-qual plan --environment $E
trtmc-aiperf-qual run --profile qwen3-0.6b-fp16 --environment $E --out out/qwen3-0.6b-fp16
trtmc-aiperf-qual run-all --environment $E --out-root out/ --shard 0/2      # host 1 of 2
trtmc-aiperf-qual summary gb300-1=nvidia@host1:/runs/out gb300-2=nvidia@host2:/runs/out \
    --ssh "ssh -J jump" --output qualification.md --html qualification.html   # remote roots over ssh
trtmc-aiperf-qual rejudge --environment $E out/*/                           # re-apply the judge, no model runs
python tools/model_benchmark.py aiperf --environment $E --aiperf-python <venv>/bin/python --out-root out/
```

`run` builds the candidate bundle when `bundle_root` does not hold it yet (trtmc-bench, in the
family environment, under the GPU lock), qualifies, and applies the bundle retention policy.
`report.md` / `report.json` in the output directory hold the verdict: `pass`, `acc-issue`,
`acc-session-state`, `acc-inconclusive`, `perf-issue` (red/yellow), `perf-inconclusive` (white),
`not-comparable`, or `error` (a phase or case failed; see "Phase errors" and `phase-errors.log`);
`build.json` records the build (`build-failed` when it failed); `report.json` carries a
reproduction command.

`run-all` runs profiles one after another (profiles sharing a checkpoint back to back), appends one
line per profile to `<out-root>/campaign.jsonl`, and skips profiles that already have a result
(`--rerun` keeps the old directory as `<profile>.<timestamp>`). `--shard INDEX/COUNT` splits the
catalog across hosts. `run-all` writes `plan.json` (every profile it must report, and configuration
errors) and exits 2 on configuration errors, 1 when a profile ended in `error` or `build-failed`,
else 0 (qualification outcomes such as `acc-issue` are results, not failures).

`summary` merges result roots: the latest run of each profile wins; planned profiles without a
result are `not-run`, configuration errors `config-error`. A remote root `[NAME=][USER@]HOST:/PATH`
is fetched over `--ssh`; local paths are read in place. `--html report.html` also writes a
self-contained, failure-first report (per model: Acc suites with failing samples, TRTMC and native
outputs side by side, Perf lights with labelled p50 values and reasons, L2, evidence links, and the
reproduction command), fetching logs and family results next to it. `--baseline ROOT` notes TRTMC
p50 regressions (> 5%) against a previous run.

### Per-machine model list

GPU memory decides which models a machine can run. `models` in its environment file selects them
for `run-all` and `plan` (an explicit `--profile` overrides it):

```yaml
models:
  include: all                    # all ready catalog profiles (default), or names / glob patterns
  exclude:                        # a reason is required
    - {profile: "flux-2-dev*", reason: "does not fit in 80 GB"}
  max_checkpoint_gib: 30          # optional: also exclude larger checkpoints (unknown sizes are kept)
```

Excluded profiles are written to `<out-root>/excluded.json` and appear in `summary` as `excluded`
with their reason, unless another result root holds a result for them.

### Disk retention

Checkpoints and bundles dominate disk use. `retention` in the environment file frees them as the
batch advances (both default to `retain`):

| key | values | effect |
|---|---|---|
| `bundle` | `retain`, `delete_on_pass`, `delete_unless_error`, `delete_built_unless_error` | delete `bundle_root/<name>/` after the model's run (`delete_built_unless_error`: only a bundle the run built itself); an `error` verdict always keeps it for the rerun |
| `hf_cache` | `retain`, `delete_unused` | `run-all` deletes a checkpoint repository from `hf_hub_cache` once no remaining profile of the batch uses it |

Deletion never leaves those roots. Reference environments and reports are kept (`rejudge`
needs only the reports). The peak is the largest single model (checkpoint plus bundle) plus the
next profile's checkpoint, which `run-all` downloads during the current run (`--no-prefetch` avoids
it).

## Extending

- New model of a known Task: nothing to add.
- New Task: one entry in `config/tasks.yaml`: its gold-labelled benchmarks (`absolute`, defined under
  `benchmarks`, scored by `gold_metrics.py` or an AIPerf grader) and the Perf output check
  (`output_grader`, a comparator in `plugins/trtmc_aiperf_plugins/accuracy.py`).
- Model needing a different input or reference option: `config/models/<profile>.yaml`.
- Model needing a differently built bundle: declare it in the family qualification case's
  `candidate.build` (used automatically), or `candidate.build` in `config/models/<profile>.yaml`.
- New machine: a new file under `config/environments/` (paths, Python interpreters, ports, lock, model list,
  retention); Docker or bare metal only differ in these paths.
