# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).

- **Acc**: the TRTMC output must match the native model's for the same inputs. A family's own
  benchmark qualification accuracy cases (dataset, native reference, metric, gate, including task
  metrics against labels such as COCO mAP or top-1 accuracy) judge its models; Task suites cover
  models without one.
- **Perf**: TRTMC must be faster than the native model (eager and, where listed, `torch.compile`)
  at the candidate's precision, timed over the whole Task call on both sides.

## Design

| Layer | What | Model-specific? |
|---|---|---|
| Execution | Two HTTP servers speaking the same `/v1/tasks/{operation}` protocol: TRTMC (`trtmc-perf-serve --backend trtmc`, later `trtmc-server`) and the native reference (`--backend reference` generic adapters, or `--backend script` for the family's declared reference). AIPerf sends every request and computes the statistics. | No |
| Family cases | `families/<family>/tests/benchmark/*.yaml`: the family's accuracy cases, its candidate build, and its performance workload request. | Family-owned |
| Task | `config/tasks.yaml`: per catalog Task, fallback Acc suites, checks for every model of the Task, and Perf settings. | No |
| Model | Derived from the catalog and the family cases. `config/models/<profile>.yaml` holds only exceptions. | Only exceptions |

`trtmc-aiperf-qual plan` prints the derived configuration of every model.

### Accuracy

**Family cases.** Each accuracy case of the family runs its own evaluation
(`qualification_tests.benchmark_qualification.accuracy.run_accuracy`: dataset, selection, native
reference, metric, gate) unchanged; only the TRTMC side is replaced: its requests go through AIPerf
to a persistent trtmc-perf-serve session. A failing case is re-run on an isolated session (a fresh
worker per request): passing there means state leaks across requests (`acc-session-state`). The
candidate bundle follows the family's candidate build (a `<profile>-qual` bundle when it differs
from the catalog manifest).

**Task suites** (models without a family case):

1. **Goldens**: the native model at the golden precision (fp32 unless the Task says otherwise),
   deterministic numerics (no TF32). Keyed by suite, reference identity (adapter and family reference
   code, checkpoint and revision), and reference platform (GPU architecture plus the reference
   environment's framework versions) under `<golden_store>/<suite>/<platform>/<key>/`;
   `publish-goldens` copies them to a shared golden store.
2. **Noise floor** (diagnostic): the native model at the candidate precision graded against the
   goldens. The declared gate always applies; a failing suite is `acc-inconclusive` only when every
   failing sample also fails for the native model at the candidate precision.
3. **Candidate**: the suites on a persistent TRTMC session, with the same isolated re-check.

| Task | Fallback suite | Grader |
|---|---|---|
| text generation | MMLU zero-shot (30 subjects, answer line); HumanEval (30, token IDs) | `parity_answer_line`, `parity_token_exact` |
| vision-language | Imagenette, three images per class, answer chosen from the ten class names | `parity_edit_distance` |
| classification | Imagenette 100 (stride over all classes) | `parity_top1` |
| encoding / embedding | STS-B 50 sentences | `parity_vector` (cosine >= 0.999) |
| image features | Beans 12 (DINOv3 kNN bank) | `parity_numeric` |
| detection | COCO 2017, 100 images | `parity_boxes` |
| segmentation / prompted segmentation | Imagenette 10 | `parity_mask` |
| transcription | LibriSpeech 30 (AIPerf dataset) + corpus WER vs labels | `parity_wer` |
| audio generation | SeedTTS English 5 around the catalog request | `parity_audio` (duration, RMS, spectrum) |
| image / video generation | PartiPrompts 3 around the catalog request (plus `clip_alignment`) | `parity_image` (geometry, thumbnail PSNR/SSIM) |
| image edit, world model, text-prompted segmentation | catalog testcase | `parity_image`, `parity_numeric` |
| reranking | BEIR SciFact 10 queries x 5 abstracts | `parity_scores` (document order) |
| time series | ETTh1 10 seeded windows (the family qualification's window) | `parity_numeric` |
| robot control, stereo, geometry | catalog testcase | `parity_numeric` (shapes exact, vectors by cosine) |

**Checks for every model of a Task** (`supplementary`): text-to-speech models also pass an ASR
round trip (`tts_intelligibility`): TRTMC and the native model speak the SeedTTS sentences, an ASR
model transcribes both, and TRTMC's word error rate must stay within 0.1 of the native model's.
Sampling models (Bark, MagpieTTS) cannot be judged per sample, but what they say can. Image and
video generation models also pass a CLIP text-alignment check (`clip_alignment`): both sides render
PartiPrompts 30 (3 for videos) at the catalog request, CLIP ViT-L/14 scores every image (8 evenly
sampled frames of a video) against its prompt, and the check fails when TRTMC's mean CLIPScore is more
than 1 point below the native model's and the paired drop is significant (one-sided 95%); videos also
keep the temporal consistency of adjacent frames (mean CLIP cosine) within 0.02 by the same rule. Pixel parity only catches gross failures, since diffusion output drifts across
precisions; the report lists each sample's scores and the two rendering directories.

A suite with `base: catalog` overrides the profile's catalog request (size, steps, seed, ...) with
its dataset fields. Exceptions stay in `config/models/<profile>.yaml`: `accuracy_source: tasks`
(Task suites despite a family case), quantization tolerance (`candidate.quantization`, the Task's
`quantized` block), `sampled: true` suites, `reference.timing_precision` (time the native model at
another precision when it is invalid at the candidate's), reference options, and build overrides.

Generated media leave the servers as digests (`trtmc_perf_serving.digests`): 64x64 thumbnails of
the first/middle/last frame and a banded log spectrum for audio, so goldens stay small.

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
Image editing, world-model generation, and text-prompted segmentation always use the family
reference, because the generic adapters would ignore their extra inputs. A `script` reference built
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
trtmc-aiperf-qual publish-goldens --source /state/golden-store --store-cli "<storage CLI>"
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
| `bundle` | `retain`, `delete_on_pass`, `delete_unless_error` | delete `bundle_root/<name>/` after the model's run; an `error` verdict always keeps it for the rerun |
| `hf_cache` | `retain`, `delete_unused` | `run-all` deletes a checkpoint repository from `hf_hub_cache` once no remaining profile of the batch uses it |

Deletion never leaves those roots. Reference environments, goldens, and reports are kept (`rejudge`
needs only the reports). The peak is the largest single model (checkpoint plus bundle) plus the
next profile's checkpoint, which `run-all` downloads during the current run (`--no-prefetch` avoids
it).

## Extending

- New model of a known Task: nothing to add.
- New Task: one entry in `config/tasks.yaml` (suite, grader, gate); add a grader to
  `plugins/trtmc_aiperf_plugins/accuracy.py` only if no existing one fits.
- Model needing a different input or reference option: `config/models/<profile>.yaml`.
- Model needing a differently built bundle: declare it in the family qualification case's
  `candidate.build` (used automatically), or `candidate.build` in `config/models/<profile>.yaml`.
- New machine: a new file under `config/environments/` (paths, Python interpreters, ports, lock, model list,
  retention); Docker or bare metal only differ in these paths.
