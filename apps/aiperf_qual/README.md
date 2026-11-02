# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).

- **Acc**: TRTMC must be as accurate as the native model on the Task's workloads. Both sides answer the
  Task's gold-labelled benchmarks (`absolute`) and each benchmark is a paired non-inferiority decision
  (`noninferiority.py`): `pass` when TRTMC's regression is shown to be below the benchmark's margin, `fail`
  when it is shown to exceed it, `inconclusive` otherwise. Tasks without a gold set compare outputs with
  the native model (conversion parity). Random-weight test models are Perf only (`accuracy_source: none`).
- **Perf**: report Native and TRTMC task-call p50 from the same benchmark responses used for Acc,
  at the recorded effective precision. Timing is a measurement, not an additional acceptance gate.
  Output-length differences, execution conditions, and partial coverage remain explicit observations.

## Design

| Layer | What | Model-specific? |
|---|---|---|
| Execution | One workload recorder, with native and TRTMC servers running separately and speaking the same `/v1/tasks/{operation}` protocol: TRTMC (`trtmc-perf-serve --backend trtmc`, later `trtmc-server`) and the native reference (`--backend reference`: the generic adapter of the operation, or a family's own pipeline, `families/<family>/reference/adapter.py`). AIPerf sends every request. | No (adapters: per family) |
| Task | `config/tasks.yaml`: per catalog Task, its benchmarks (`absolute`) and checks (`supplementary`), and Perf settings. | No |
| Model | Derived from the catalog entry and its Task. `families/<family>/tests/aiperf/<profile>.yaml` owns exceptions and benchmark definitions; legacy `config/models/<profile>.yaml` remains supported. | Per family |

`trtmc-aiperf-qual plan` prints the derived configuration of every model; `trtmc-aiperf-qual matrix`
writes the execution matrix (native path, environment, workloads, checks, and whether each profile is
executable).

Family definitions are merged after Task defaults and legacy overrides. They may declare local
`benchmarks` and select them through `absolute`, so model-specific requests, native adapters and
conversion criteria stay with their family. RT-DETR v2 keeps the bundle's fixed score threshold of
0.3 on both sides. BiRefNet Lite compares binary foreground masks on 50 pinned COCO images against
its official native model, using the unchanged family E2E IoU threshold of 0.90 for every image.
This is conversion parity, not ADE20K semantic-segmentation accuracy. Acc and task-call P50 use the
same requests. Remote-code module caches default to `hf-modules` beside the configured Hub cache;
an explicit `HF_MODULES_CACHE` takes precedence.

### Accuracy

Benchmarks (`benchmarks` in `config/tasks.yaml`) are AIPerf accuracy benchmarks at pinned revisions
(MMLU 0-shot, LAMBADA, TinyStories, BART denoising: `plugins/trtmc_aiperf_plugins/benchmarks.py`) or gold
suites sent through AIPerf's `trtmc_task` endpoint and scored by `gold_metrics.py` (MMStar, OCRBench,
RefCOCO, LibriSpeech WER, STS-B, SciFact retrieval / rerank, HumanEval + MBPP, ImageNetV2, COCO mAP,
ADE20K mIoU, mask IoU, ETTh1 MSE, WMT / FLORES chrF++). Selections are seeded and stratified; prompts are
filtered to the shipped bundle's length (rendered through the chat template on the chat route). A model
whose catalog request samples answers once per seed on each side and is judged on per-problem seed means.
Each benchmark's `margin`, `relative_margin`, and size are set in `config/tasks.yaml`.
Native absolute scores do not decide conversion acceptance: the paired difference does.
A low score limits what this benchmark establishes about model capability, but does not
prevent comparing conversion results. Historical `min_native` fields are ignored.
Equal zero task scores establish only the scored benchmark comparison, not identical outputs
or intrinsic model capability. This applies to the random-weight DeepSeek V2 tiny fixture too.

Conversion-parity benchmarks (`gold_metrics.PARITY`) compare each output with the native one within a
tolerance: raw encoders' vectors, MoGe geometry, ACT action chunks, stereo disparities, PersonaPlex speech.
Inputs a family prepares itself come from its `reference/inputs.py` (`family_inputs` suites).

A family configuration can declare a complete local `benchmarks` entry (`suite`,
`suite_definition`, `metric`, and `gate`) and select it through `absolute`, without adding
family policy to the shared Task defaults. `family_parity` delegates
to `gate.checker`, a `families.<family>.<module>:<function>` taking the two observations and gate;
the ordinary parity decision still requires every selected sample to pass.
DINOv3 uses 50 seeded COCO val2017 images, complete hidden and pooled features, and its existing
family E2E criteria. Accuracy and task-call P50 come from the same extraction requests.

Checks for every model of a Task (`supplementary`): text-to-speech round-trip WER (corpus, bootstrap) and
audio validity (every output finite and not silent; the median per-sentence duration ratio to the native model
within 0.5-2, as a sampling model's single utterance may run to its length limit on either side; an error
when no sentence has audio on both sides); GenEval-style pass rate for text-to-image families that take caller latents (the same
initial noise on both sides); CLIP-T and video validity for videos; MagicBrush CLIP-I and DINO for edits;
world-model video parity. The pixel parity under latent replay is reported only (`informational`).

A suite with `base: catalog` overrides the profile's catalog request with its dataset fields.
For a profile with `candidate.benchmark_sequence_length`, the build counts the prepared
gold-suite prompts with its pinned tokenizer and uses their maximum length plus the configured
`extra_tokens`. Qwen3-Embedding reserves one token for EOS and four for runtime tokenizer count differences.
The evidence is written to
`artifacts/build/benchmark-capacity.json`; request truncation and accuracy tolerances do not change.
Dataset or tokenizer failures stop capacity planning rather than falling back to the catalog size.
Lance uses the `Lance_3B` tokenizer subfolder and reserves its image, template, and generation tokens.
Its Native adapter loads the pinned upstream model once and serves the same MMStar requests at the
public Task's 448x448 image size, with image decode before timing and preparation/generation inside it.

### Verdict rules

- Both sides answer the same problems, selected and fitted to the shipped bundle's sequence length; a suite with no
  problem that fits is an `error` that says so.
- An input TRTMC rejects as beyond the bundle's capacity (`backend_rejected_request` exceeding a prefill profile, a
  KV cache, or an input limit) leaves the comparison on both sides, and the entry reports how many did
  (`out_of_capacity`); not in a corpus whose rows refer to each other (STS pairs, a retrieval query and its
  documents).
- Any other problem the native model answered and TRTMC did not (a failed or rejected request, an unusable output)
  counts as TRTMC's wrong answer: a wrong right/wrong answer, a parity sample outside the tolerance, an empty WER or
  chrF text. A corpus metric without an empty answer (vectors, masks, detections, forecasts) keeps the problem
  missing, an `error`, as does a problem the native model did not answer.
- A parity benchmark against a native model that ran at another precision than TRTMC (the candidate's failed
  natively) uses its `mismatched_precision_gate`.
- A native precision whose requests all fail tries the configured fallback precision and preserves the failed
  attempt. Low scores from successfully returned answers do not trigger precision fallback.
- Both sides run separately with one server and concurrency 1. Legacy Acc replica, MPS, and overlap settings
  are ignored by qualification; their contended timings cannot serve as speed evidence.
- Every AIPerf run has a 12-hour deadline unless the environment explicitly sets `deadlines.<profile>`.
  Scheduling ledgers only determine assignment: historical rescoring durations can omit fresh inference.
  A GPU phase that fails before producing its result runs once more.
- `summary` reports one result per model, worst first: White (no verdict: an error or a failed build; or no valid
  comparison: a Task without an Acc check), Red (Acc worse than
  native beyond its margin), Yellow (an Acc difference not shown either way), Green (quality passes with available
  benchmark timings). Historical fixed-workload reports retain their original performance lights.
  Profiles excluded by the machine model list remain in `excluded.json` for audit and do not contribute
  report rows, totals, White counts, or pass-rate denominators. A measured result from another host still counts.
  Catalog profiles without Task test defaults remain in `plan.json` for coverage audit, outside those
  statistics too. Invalid configurations of supported tests still appear as White errors.

### Performance

One executor sends every workload through AIPerf and writes one `execution.jsonl` evidence stream.
Workloads have a role: `quality`, `performance`, `both`, or optional `service`. Both consumers reuse
profiling responses and artifact references; image and video quality scoring does not regenerate outputs
just to obtain their task-call timings. Warmup is excluded, both backends run separately at concurrency 1,
and GPU scorers run after their generation servers stop. This costs more wall time than the former
multi-replica, overlapping Acc schedule.
Generated artifacts are joined to profiling requests by request identity and payload, then ordered
by the suite inputs. Warmup advances the dataset cursor; arrival order must never select the gold label.
The GPU probe selects the configured device and waits briefly for startup activity to settle.
A formal benchmark cannot start if utilization remains at least 20% or the probe fails.

The report answers two independent questions: the task quality scores on each side and their difference;
and the native/TRTMC task-call times, precision, and work comparability. Markdown and HTML omit speedup
ratios; JSON retains them and their intervals for qualification and diagnostics. Quality uses
its existing task metric and non-inferiority gate. Parity is labelled parity, random-weight models have
absolute quality N/A, and an unavailable native reference cannot provide a speedup. CLIP-T and video
validity do not establish temporal, motion, or action correctness; world-model checks establish coarse
conversion parity only. The family's checks keep their documented coverage and thresholds.

Natural evaluation workloads (`both`) provide timings from the **same outputs used for quality**. Their
paired geometric speedup and total-time ratio are descriptive; the 90% interval is across dataset units,
with seeds clustered by problem, not a repeated-run stability interval. The report shows each side's p50,
effective precision, successful timing coverage, and observed work differences.
Generation length and work comparability do not decide whether benchmark timings are measured.
Summary rows flag unequal or unknown work and partial timing coverage beside the P50 values.
Inputs excluded by Acc as exceeding bundle capacity are excluded from both timing sides too; their
count and the original attempted request counts remain visible. Other failed or missing requests
remain in coverage and produce a `partial` timing measurement when both sides have timings.
A missing workload or a side without any valid timing is an error. `max_tokens` alone is not work evidence.
No mandatory second, forced-length suite is added for variable-output families.
GenEval now defaults to `geneval-full`, all 553 prompts at the pinned metadata hash;
the old `geneval-200` preset remains available for an explicitly smaller selection.

Models with quality benchmarks run **only their required quality workloads**. For example, Qwen uses
`mmlu-0shot`, and image/video models use their configured quality datasets. Catalog, near-capacity,
and informational replay checks are not extra default workloads. Each side answers each selected
problem once, with excluded warmup. The same profiling responses supply Acc, Native/TRTMC task-call
p50, and AIPerf client metrics. Multiple required benchmarks remain separate, labelled datasets.
Dataset timing completeness and work comparability are recorded as observations; timing results
are measurements, not repeated-run performance acceptance gates. Quality thresholds remain unchanged. Failed or
unpaired responses remain visible in each side's timing coverage rather than disappearing into a
matched subset. Models explicitly lacking a quality benchmark retain one configured performance
workload and conversion-parity evidence. The explicit `order-check` diagnostic and historical
fixed-workload reports keep their original statistics. Timing and generation phases hold `gpu_lock`.

Configuration now uses a flat `performance` policy and opt-in `service_metrics`; reports use `performance`
and `service_metrics` under schema `trtmc.qualification/v2`. Earlier tiered configurations and reports
are normalized on read, including rejudge, without maintaining another execution path. Rejudge without an
environment refreshes benchmark timings from saved execution records while
preserving the recorded Acc entries and gates. It never promotes descriptive dataset results to
an acceptance gate. `torch.compile` remains an optional labelled
reference. Service metrics (client latency, throughput, load sweeps) are opt-in and do not affect the
verdict; the prototype's single execution lane and buffered SSE do not measure token TTFT/ITL.

Every recorded workload also includes AIPerf's native client statistics in `aiperf_metrics`: request
latency p50/p99, request throughput, output-token throughput when available, and request error rate.
JSON retains every run's exported statistics and units. Markdown and HTML show one Native/TRTMC table
per workload, with separate precision, run-count, and total-request columns. Repeated runs use the median
of each exported statistic; displayed latency p50/p99 are medians of run percentiles, not pooled request
percentiles. Hovering a metric in HTML shows the range across runs. Different workloads, modes, precisions,
and concurrency levels are never averaged together. The main Native row uses the declared timing precision
when available; additional native settings (such as fp32 quality references or compiled modes) appear in
a separate collapsed section. Precision mismatches remain explicit. Output-token throughput appears only
where exported; missing metrics stay unavailable and partial statistics show available/total runs.
HTML shows these tables directly below the model summary, with links from each model and the same
search/result filters. Summaries reuse existing exports without extra inference or load sweeps and never
affect acceptance gates or the model verdict.

```yaml
performance:
  measurement: {settle_s: 10, warmup: 3, requests: 12, runs: 5, min_run_s: 1.0}
# Optional for a service workload, e.g. LLM/VLM:
service_metrics: {isl: 96, osl: 32, concurrency: [1, 4], requests: 32}
```

### Reference environments

A model whose native reference needs more than the serving interpreter declares
`reference.requirements` (normally `families/<family>/requirements.txt`, and `reference.prepare` for an
upstream checkout) in `config/models`; `trtmc-perf-serve reference-env` creates the environment layered on
the serving interpreter once, caches it by digest, and creates a fresh one next to it when its recorded
`pip freeze` changed. Bundles build in the same environment, as CI installs a family's requirements before a
build; trtmc-bench reuses a bundle only when its build receipt matches.

The Diffusers adapter ties a T5-style text encoder's `encoder.embed_tokens` back to `shared` when loading
left it zero (Transformers 5.2 does not tie it for `UMT5EncoderModel`, Wan). A timed request that leaves
`num_steps`, guidance, CFG, or frames at -1 is an error until `config/models` states them, since TRTMC and
Diffusers apply different defaults.

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
trtmc-aiperf-qual matrix --environment $E --output matrix.csv                # exit 1 unless every profile is executable
trtmc-aiperf-qual run --profile qwen3-0.6b-fp16 --environment $E --out out/qwen3-0.6b-fp16
trtmc-aiperf-qual run-all --environment $E --out-root out/ --shard 0/2      # host 1 of 2
trtmc-aiperf-qual summary gb300-1=user@host1:/path/to/out gb300-2=user@host2:/path/to/out \
    --ssh "ssh -J jump" --output qualification.md --html qualification.html   # remote roots over ssh
trtmc-aiperf-qual rejudge --environment $E out/*/                           # re-apply the judge, no model runs
python tools/model_benchmark.py aiperf --environment $E --aiperf-python <venv>/bin/python --out-root out/
```

`run` builds the candidate bundle when `bundle_root` does not hold it yet (trtmc-bench, in the
reference environment, under the GPU lock), qualifies, and applies the bundle retention policy.
`report.md` / `report.json` in the output directory hold the verdict: `pass`, `acc-issue`,
`not-covered` (no native path), `acc-inconclusive`, `not-comparable`, `perf-issue` (red/yellow),
`perf-inconclusive` (white), or `error` (a phase failed; see "Phase errors" and `phase-errors.log`);
`build.json` records the build (`build-failed` when it failed); `report.json` carries a reproduction
command. `--smoke` runs one problem per benchmark and one timed request (`smoke-pass` / `smoke-fail`,
never a verdict; results go under `<out-root>/smoke`, or `<dir>/smoke/<profile>` for `run --out <dir>/<profile>`).
`rejudge` and `recheck` keep the run's own report as `report.original.json`.

AIPerf 0.13.0 restarts its session counter after warmup while continuing the dataset sampler.
The startup hook installed by `doctor --fix` grades by `conversation_id` and preserves the
original phase/session metadata. After upgrading the plugins, run `doctor --fix` again to
update an existing hook. Qualification pairs answers and timings by the exported dataset identities.

To recover earlier plugin benchmark scores without model inference, use
`trtmc-aiperf-qual rejudge --selection-cache /path/to/hf-datasets/trtmc-accuracy-selections out/*/`.
Recovery requires the original selection cache, `inputs.json`, accuracy exports, raw exports,
and execution records. It verifies the complete ordered inputs and both sides' gold selections,
uses the recorded graders and gates, and retains the original response exports and report.
Corrected evidence is written to `accuracy_export.aligned.jsonl` and `execution.aligned.jsonl`.
Missing or conflicting evidence fails recovery instead of producing an inferred score.

`run` exits 0 for `pass`, `smoke-pass`, or `measured` with Acc `pass` and Perf `measured`; other
verdicts exit 1. A complete benchmark measurement keeps its `measured` report category, without
claiming repeated-run performance acceptance.

`run-all` runs profiles one after another (profiles sharing a checkpoint back to back), appends one
line per profile to `<out-root>/campaign.jsonl`, and skips profiles whose finished result has the same
run key (configuration, harness, mode, and dependencies; `--rerun` keeps the old directory as
`<profile>.<timestamp>`). `--shard INDEX/COUNT` splits the
catalog across hosts. `run-all` writes `plan.json` (every profile it must report, and configuration
errors) and exits 2 on configuration errors, 1 when a profile ended in `error` or `build-failed`,
else 0 (qualification outcomes such as `acc-issue` are results, not failures).

`summary` merges result roots: the latest run of each profile wins; planned profiles without a
result are `not-run`, configuration errors `config-error`. A remote root `[NAME=][USER@]HOST:/PATH`
is fetched over `--ssh`; local paths are read in place. `--html report.html` also writes a
self-contained, failure-first report (per model: Acc suites with failing samples, TRTMC and native
outputs side by side, Perf lights with labelled p50 values and reasons, L2, evidence links, and the
reproduction command), fetching logs next to it. `--baseline ROOT` notes TRTMC
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
batch advances (all default to `retain`):

| key | values | effect |
|---|---|---|
| `bundle` | `retain`, `delete_on_pass`, `delete_unless_error`, `delete_built_unless_error` | delete `bundle_root/<name>/` after the model's run (`delete_built_unless_error`: only a bundle the run built itself); an `error` verdict always keeps it for the rerun |
| `temporary_files` | `retain`, `delete_on_pass`, `delete_unless_error` | after scoring and server shutdown, remove generated request `scratch/` directories (images, videos, audio, arrays, uploaded inputs and initial noise); keep raw exports, scores, timings and logs |
| `hf_cache` | `retain`, `delete_unused` | `run-all` deletes a checkpoint repository from `hf_hub_cache` once no remaining profile of the batch uses it |

`delete_on_pass` also accepts a benchmark result with passing Acc and complete Perf (`measured`).
Partial timings, inconclusive Acc, and failed Acc keep their bundles under that policy.
Failed Native precision attempts remain in the execution evidence; their timings and client
statistics are excluded when a fallback succeeds.

Use a cache and bundle root dedicated to this campaign when enabling deletion: `delete_unused`
means unused by the selected batch, not by unrelated processes or campaigns sharing that cache.
HF cache deletion removes raw checkpoint weights, so it is opt-in; the default keeps them.
Even with `delete_unused`, a checkpoint needed by any profile with a framework/build error is kept.
Build and framework errors keep bundles and temporary files. `retention.json` records the policies,
removed paths and recovered bytes; checkpoint deletion is also recorded in `campaign.jsonl`.

For disk-constrained runs that keep raw weights:

```yaml
retention:
  bundle: delete_built_unless_error
  hf_cache: retain
  temporary_files: delete_unless_error
```

Only set `hf_cache: delete_unused` for a cache this batch exclusively owns. Deletion never leaves
the configured roots. Reference environments, datasets (with the shared media store), runtime binaries and reports are kept.
The peak includes the current checkpoint, bundle and generated media, plus the next checkpoint
if prefetch is enabled (`--no-prefetch` avoids that overlap). Temporary cleanup happens per model,
so there must still be enough space for one model's complete scoring workload.

Each new model result has one working directory:

```text
<profile>/
  report.md / report.json       # Acc and Perf together
  model.json                   # resolved configuration
  execution.jsonl              # requests, identities, timings and work evidence
  build.json / retention.json  # build identity and cleanup receipt
  run-key.txt                  # resume identity
  artifacts/                   # build logs, AIPerf exports, server logs and score details
    build/
    <workload-and-side>/        # scratch/ exists only when retained
```

An error may additionally write `error.json` or `phase-errors.log`. Older output layouts stay
readable; reruns use the new layout and preserve prior results separately. Raw AIPerf references
in new execution records are relative to the model result, so the entire result can be moved
and raw-response timing recovery still works with the same request hash and sample checks.
Accuracy regrading also needs its original selection cache. Reading reports and rejudging stored
scores/timings do not require bundles or HF weights. After temporary cleanup, media rescoring
requires generation again; text raw responses remain available. If the bundle is also gone,
`recheck` fails before replacing valid scores: run the model again to rebuild and regenerate. Cleanup does not promise
bit-for-bit model replay without the checkpoint, bundle and generated artifacts.

Media rechecks preserve the original exports. A regenerated side writes under its existing workload
directory's `rerun-<timestamp>/`; its responses replace that side's Acc, task timings and informational
AIPerf metrics together. `execution.rechecked.jsonl` keeps the active measurements and superseded history.
Outputs are reused only with their matching execution evidence; a failed regeneration cannot retain an
older passing timing. Workloads outside the recheck keep their original measurements.

### Shared media store

A dataset with images, audio or video (ImageNet, COCO, LibriSpeech, ...) is kept **once per machine**, not once
per model. `build_suite` writes every input to a content-addressed store and the request carries only its digest:

```text
<media_root>/<aa>/<sha256><suffix>     # read-only, named by content; one store for every model, side, suite and run
```

```json
{"image_path": {"$file": {"suffix": ".jpg", "sha256": "<64 hex digits>"}}}
```

The store defaults to `<hf_datasets_cache>/trtmc-media`; `media_root` in the environment file moves it (put it on the
large disk). `services.serving` starts every server with `--media-root`; a server resolves a digest only inside its own
store, checks the bytes against it, and hands the backend a per-request copy, so a client never names a path and a
backend never modifies the store. Because requests no longer contain the bytes, `<suite>.inputs.jsonl`, AIPerf's
`inputs.json` and `profile_export_raw.jsonl` hold a few hundred bytes per request instead of the whole image: a
10 000-image ImageNet suite wrote about 5 GB of inline base64 per side and model, and now writes kilobytes per request.

`request_sha` is content-bound (the same bytes have the same digest on every machine), so a result stays
self-consistent without the store. Inline `b64` remains accepted by the server for ad-hoc requests and the OpenAI routes;
results written before this change keep their inline payloads and stay readable. Any run of a suite repopulates the store
from its dataset, so the store is a cache: `retention` never deletes it, and deleting it only costs the next run a rewrite.
A suite with media and neither `media_root` nor `hf_datasets_cache` configured is a configuration error.

## Extending

- New model of a known Task: nothing to add.
- New Task: one entry in `config/tasks.yaml`: its gold-labelled benchmarks (`absolute`, defined under
  `benchmarks`, scored by `gold_metrics.py` or an AIPerf grader) and the Perf output check
  (`output_grader`, a comparator in `plugins/trtmc_aiperf_plugins/accuracy.py`).
- Model needing a different input or reference option: `config/models/<profile>.yaml`.
- Model whose native pipeline the generic adapters cannot run: `families/<family>/reference/adapter.py`
  (an `Adapter(spec, host)` with `invoke(request, artifact_base)`; it imports nothing from the applications and
  reaches the serving mechanics through `host`), named by `reference.adapter`.
- Model needing a differently built bundle: `candidate.build` (or `candidate.model_directory`) in
  `config/models/<profile>.yaml`; the report names the bundle it qualified.
- New machine: a new file under `config/environments/` (paths, Python interpreters, ports, lock, model list,
  retention); Docker or bare metal only differ in these paths. A run's own inputs (its ledger and multi-host
  assignment) stay with that run's results and are passed by path (`run-all --ledger`, `--assignment`).


## Native endpoint group coverage

The native `trtmc-server` endpoint expansion is tracked in
[epic #1667](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/1667).
Stages 1–10 require every supported checkpoint in each group for one workload.
Use each stage issue for its per-model results. The original pilot results
below qualify only their named profiles; their pilot PRs alone do not establish
full-group coverage.

Discover the current checkpoint inventory from family-owned manifests:

```bash
PYTHONPATH=apps/aiperf_qual:apps/benchmark:core/builder:. \
  python -m trtmc_aiperf_qual endpoint-models --repo . --group embeddings
```

Groups are `embeddings`, `reranking`, `transcription`, `detection`, `vision`,
`images`, and `image-edit`.
The image group excludes legacy video profiles even when their Task string is
`image_generation`.

For an already-built image bundle that passed its owning family's unchanged
correctness checks, use `image-profile` with the same environment file shown
below. Inputs are AIPerf single-turn JSONL (`{"text":"a prompt"}`); catalog
dimensions, steps, guidance and seed are applied explicitly. Per-row `extra`
may carry supported controls or `initial_latents_b64` in the family's existing
raw float32 layout for matched-noise replay.

```bash
PYTHONPATH=apps/aiperf_qual:apps/benchmark:apps/perf_serving:core/builder:. \
  python -m trtmc_aiperf_qual image-profile \
    --environment /path/to/local-environment.yaml \
    --profile stable-diffusion-v1-5 --bundle /path/to/stable-diffusion-v1-5.bundle \
    --inputs /path/to/image-prompts.jsonl --request-count 5 --warmup-request-count 1 \
    --out /path/to/new-image-results
```

This uses pip-installed AIPerf `image_generation` through `trtmc-server`.
It joins every client response and metric to its native request record, checks
the complete decoded PNG against that invocation's native artifact SHA-256,
and saves measured and warmup images separately by request identity in
`images.json`. `summary.json` separates HTTP latency, Task time and images/sec.
Transport identity does not replace the family's model-quality gate.
The command checks the bundle's recorded checkpoint, revision and build controls
against the selected existing profile, and retains the actual build provenance.
Use a current-builder bundle with that provenance for qualification.
See [Profile Image Generation with AIPerf](../../website/docs/user-guides/profile-images-with-aiperf.md)
for the three-step interactive example.

For one reference image plus instruction, use `image-edit-profile` instead:

```bash
PYTHONPATH=apps/aiperf_qual:apps/benchmark:apps/perf_serving:core/builder:. \
  python -m trtmc_aiperf_qual image-edit-profile \
    --environment /path/to/local-environment.yaml \
    --profile qwen-image-edit-2511 --bundle /path/to/qwen-image-edit-2511.bundle \
    --inputs /path/to/image-edits.jsonl --request-count 5 --warmup-request-count 1 \
    --out /path/to/new-edit-results
```

Each JSONL row supplies `text`, one local `image` path (relative to the dataset
file or absolute) or an embedded image data URL, and optional `extra` controls.
The qualifier pins decoded RGB pixels in lossless PNG uploads, preserving the
original source hash and avoiding AIPerf's lossy re-encoding of local JPEGs.
It reuses the image-generation server/qualifier, with AIPerf `image_edit` and
multipart `/v1/images/edits`. Each input image/instruction pair, native controls,
replay noise and complete edited output is checked by request identity. Saved
inputs accompany their outputs in `images.json`. Build geometry is recorded
separately from actual conditioning and requested output geometry: an edit
builder may use its image dimensions for the input. Checkpoint, Task, precision
and other build provenance checks remain mandatory. Quality still requires the
owning family's unchanged correctness criteria on the exact served bundle.
See [Profile Image Editing with AIPerf](../../website/docs/user-guides/profile-image-edits-with-aiperf.md)
for the three-step Qwen example.

The inventory selects a normal full-model profile when one is available, with
precision and parallel variants listed separately. Checkpoints without one
remain explicit gaps. Discovery reports `unverified`; it does not claim
correctness or AIPerf qualification. An encoder bundle is not counted as a reranker merely because
the family implementation inherits both interfaces. Streaming-oriented ASR
checkpoints with a whole-clip Task belong to the recorded-clip workload; this
does not establish streaming HTTP support.

After running the owning family's unchanged correctness checks on an actual
bundle, `endpoint-profile` automates the existing native/HTTP and pip-client
checks for that profile. It reuses `trtmc-server`, the native benchmark worker,
the published AIPerf runner, and request/metric joins. Use a local environment
file containing your paths:

```yaml
repo: /path/to/TensorRT-Model-Connect
serve_python: /path/to/serving-python
runtime_root: /path/to/native-build
aiperf_python: /path/to/aiperf-venv/bin/python
aiperf: /path/to/aiperf-venv/bin/python
hf_datasets_cache: /path/to/datasets
```

```bash
PYTHONPATH=apps/aiperf_qual:apps/benchmark:apps/perf_serving:core/builder:. \
  python -m trtmc_aiperf_qual endpoint-profile \
    --environment /path/to/local-environment.yaml \
    --profile bge-small-en-v1.5 --bundle /path/to/bge.bundle \
    --inputs apps/aiperf_qual/examples/embedding-inputs.jsonl \
    --out /path/to/new-result-directory
```

The default client tokenizer uses the profile's pinned checkpoint revision.
Use `--tokenizer /path/to/prepared-tokenizer` when its files live separately;
`--tokenizer-revision` pins an explicitly selected Hugging Face tokenizer.
Hub overrides require a full commit SHA. Local tokenizer directories are
fingerprinted before and after profiling, including their file contents.
Lance and Phi4 provide `families/<family>/tests/aiperf/prepare_tokenizer.py`
to export their existing tokenizers into a standalone client directory.
Source identity is captured after acquiring the GPU lock, including content
fingerprints for tracked changes and untracked files. Generated run output is
excluded. Changes to source, tokenizer, bundle or inputs between the snapshots
preserve raw evidence and prevent a success summary.

Use the existing sentence/ranking examples or prepared audio/image JSONL from
the guides below. The command verifies every corpus input against native output,
then checks all 55 complete responses from five warmups and 50 measured requests,
including SHA-256 hashes of media bytes, complete vectors/transcripts/boxes/answers,
native class IDs or passage indices, and every request/timing join. Results
separate HTTP latency from public Task time. Build/load/startup, native/HTTP verification and
warmup are excluded from measured throughput. This transport verification is
separate from the owning family's numerical/quality criteria. It establishes
no capacity or performance regression gate. The client is AIPerf 0.13.0 from
PyPI; an AIPerf source checkout is unnecessary.

DeepSeek-OCR-2 preserves aspect ratio with mean-color padding in its existing
768-pixel, 145-visual-token single-view profile. Rebuild older square-warping
bundles with the current family runtime. Its unchanged owning reference uses
1024-pixel crop-mode inference; record that different visual budget when
interpreting conversion parity. This endpoint run measures the native Task
and does not claim a reference-relative speedup.

Native InternVL3 retains one 448-pixel, 256-visual-token view. Its unchanged
owning reference may tile one image into multiple views. Record that visual
budget difference alongside conversion parity; serving one original image
does not establish identical internal view counts.

## Native endpoint pilot coverage

These original deliveries are historical pilot evidence. Use the stage issues
for the expanded per-model correctness and client results.

| Profile | Endpoint / AIPerf type | Qualified workload | Evidence |
| --- | --- | --- | --- |
| `bge-small-en-v1.5` | `/v1/embeddings` / `embeddings` | One sentence per request, FP16, context 512, concurrency 1; five fixed example sentences, 5 warmups and 50 measured requests | [Stage 1](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/1668) |
| `nemotron-rerank-vl-1b-v2` | `/v1/ranking` / `nim_rankings` | One query plus two fixed text passages, FP16, context 685, concurrency 1; 5 warmups and 50 measured requests | [Stage 2](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/1669) |
| `whisper-tiny-fp16` | `/v1/audio/transcriptions` / `audio_transcription` | One complete mono PCM16 WAV clip; ten pinned LibriSpeech clips, FP16, context 128, output limit 120, concurrency 1; 5 warmups and 50 measured requests | [Stage 3](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/1670) |
| `yolo11n` | `/v1/infer` / `image_retrieval` | One original RGB image per request; 100 pinned COCO images, FP16, family threshold 0.25, concurrency 1; 5 warmups and 50 measured requests | [Stage 4](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/1671) |

The embedding pilot passed the unchanged bert family correctness checks and a
real published AIPerf 0.13.0 run on one GB300 with TensorRT 11.1.0.106. All
55 responses contained complete 384-value vectors.

The reranking pilot passed the unchanged eagle_vlm family correctness checks
and the same published client run configuration. All 55 responses retained
real finite scores and original passage indices. Both pilots matched all
client/server request IDs and timing exports. Additional epic stages remain
planned until their native correctness and actual client runs are complete.

The transcription pilot passed the existing whisper family native/reference
testcase and benchmark WER criteria on the exact served bundle. The ten native
transcripts matched the FP32 reference under family WER normalization; both
had 3.59% WER against the gold corpus. All 55 published-client responses matched
the corresponding native transcript, with complete request and timing joins.
Audio sample rates, counts, and durations remain recorded. This stage qualifies
recorded complete clips; streaming ASR and audio question answering remain planned.
Whisper now honors the checkpoint GELU with an FP32 activation boundary for
half inputs, and an FP32 encoder disables TF32 operand rounding. The decoder
computes its comparison-sensitive output projection in FP32. Older Whisper
bundles need rebuilding. The numerical gates remain unchanged.

The detection pilot passed the unchanged yolo11 checkpoint/native/reference testcase
and 100-image family mAP gates. Native and reference mAP@[.50:.95] were both
about 39.5% (permitted drop: 2 percentage points). One grayscale
image was skipped under the RGB input contract and remains in the selection audit.
All 100 HTTP/native verification calls and 55 published-client responses preserved
the complete boxes, native class IDs, scores, and coordinate normalization; request
and timing joins were complete. Prepared data URLs preserve original image bytes
through AIPerf. This qualifies object detection; document/table layout remains planned.

Use [Profile Embeddings with AIPerf](../../website/docs/user-guides/profile-embeddings-with-aiperf.md),
[Profile Reranking with AIPerf](../../website/docs/user-guides/profile-reranking-with-aiperf.md),
[Profile Transcription with AIPerf](../../website/docs/user-guides/profile-transcription-with-aiperf.md),
[Profile Object Detection with AIPerf](../../website/docs/user-guides/profile-detection-with-aiperf.md),
or [Profile Image Understanding with AIPerf](../../website/docs/user-guides/profile-vision-with-aiperf.md)
for the short recipes. Native Task-call wall time stays separate from client
HTTP latency; these small runs are integration examples, not performance gates.

## Text profiling with the persistent server

`text-profile` reuses `trtmc-server` with AIPerf's built-in `completions` and
`chat` endpoints. It adds server readiness/model checks, AIPerf release
version checks and installed-package provenance, family validation commands,
length/load sweeps, request-ID timing joins and an optional sequential
native-reference comparison. It produces
benchmark evidence independently of the qualification verdicts above.

Qwen and GPT-2 prebuilt validation reads the bundle's core build provenance with
`trtmc inspect` and checks the recorded immutable Hugging Face checkpoint identity and
build settings against the selected pinned snapshot and family manifest. Bundles built
without core build provenance must be rebuilt for automated exact-bundle validation.
Declaring a profile or revision in YAML is insufficient.


```bash
PYTHONPATH=apps/aiperf_qual python3 -m trtmc_aiperf_qual text-profile \
  --environment /path/text-environment.yaml \
  --config /path/qwen-text-profile.yaml --out /path/results/qwen-baseline
```

For a short guide using an already-built Qwen bundle and the pip-installed
AIPerf client directly, see
[Profile Text with AIPerf](../../website/docs/user-guides/profile-text-with-aiperf.md).
The automated runner also uses `pip install aiperf==0.13.0`; no AIPerf source
checkout is required. Keep the client in a separate Python environment.

For automatic validation and timing joins, generate the configuration from
the checked-in Qwen example, then run `text-profile`. Replace the three paths
below with your native build, already-built FP16 bundle, and client interpreter.
The Qwen example expects checkpoint revision
`c1899de289a04d12100db370d81485cdf75e47ca` and a context limit of 256.
Use your CUDA-enabled TRTMC Python interpreter for both commands; the runtime
build must also include `qwen_text_stream_consumer` for the family stream check.

```bash
python apps/aiperf_qual/examples/prepare_text_profile.py \
  --build-dir /absolute/path/build \
  --bundle /absolute/path/qwen3-0.6b.bundle \
  --aiperf-python /absolute/path/client-venv/bin/python \
  --out artifacts/qwen-config

PYTHONPATH=apps/aiperf_qual python -m trtmc_aiperf_qual text-profile \
  --environment artifacts/qwen-config/environment.yaml \
  --config artifacts/qwen-config/model.yaml --out artifacts/qwen-profile
```

Run in the same GPU environment as the build, with the reference checkpoint
available in its Hugging Face cache. The runner validates the exact served
bundle before timing. `report.json` contains success counts, p50/p95 latency,
throughput and native Task wall time; `run-NNN/joined.jsonl` joins client
measurements to server records. Use new configuration and result directories
for subsequent runs. Add `--include-streaming` to generate Qwen's raw/chat
streaming cases. Larger bundles can need a higher `server.startup_timeout`.

To profile another built model with this runner, supply a model template with
its checkpoint/tokenizer revisions, server settings and family-owned
`validation_commands`. These checks must validate the exact served bundle.
Do not reuse Qwen's correctness commands for another family. Optional legacy
`aiperf.source` and `aiperf.commit` fields retain exact clean-checkout checks
for source installations, but the shipped templates use `aiperf.version` only.

In Docker, make the checkout, build, bundle, client environment and cache
available inside the GPU container. For a Git worktree, mount the original
checkout's entire `.git` directory at the same absolute path too:
`git rev-parse --path-format=absolute --git-common-dir` identifies it. The
automated runner reads Git metadata to record provenance.

Client token counts are tokenizer estimates; do not enable
`--use-server-token-count`, because the server does not expose measured prompt
token counts. Nonstreaming native `model_call_ms` measures the public Task
call; streaming measures the Task stream including relay/backpressure.
AIPerf `client_request_latency_ms` ends at the last content chunk;
`client_request_lifecycle_ms` also includes terminal delivery and cleanup.
These serving measurements exclude model loading, validation and warmup and
do not represent GPU kernel time or release qualification verdicts.

An optional `reference` block (`profile`, GPU `python`, `mode: eager`) runs
the existing native reference sequentially under the same GPU lock. Only
nonstreaming, closed-loop, concurrency-one cases with matching inputs, text,
actual output counts and Task timing scopes receive a descriptive native Task
wall ratio. Mismatched work is reported as noncomparable.

The normal qualification commands keep the existing perf-serving service and
its raw Task/observation protocol. `services.serving(service="trtmc-server")`
is an explicit text candidate option; callers must use the public text endpoints.

### Complete text-to-video clips

Use the existing full `minimax-h3-768p`, `wan21-t2v-1.3b` or `wan22-ti2v-5b`
profile after its unchanged family correctness case passes on the exact bundle.
`endpoint-models --repo . --group videos` also reports partial/L0-only and
distributed variants; inventory is not qualification. No new model specs are needed.

```bash
python -m trtmc_aiperf_qual video-profile \
  --environment environment.yaml --profile wan21-t2v-1.3b \
  --bundle /path/to/validated.bundle --inputs prompts.jsonl \
  --out artifacts/wan21-video --request-count 2 --warmup-request-count 1
```

Each input row has `text` and optional `extra` controls, for example
`{"text":"A cat sitting on a beach watching the sunset"}`. Geometry, steps, seed
and declared guidance come from the existing profile; omitted guidance keeps
the public Task's default. `extra.initial_latents_b64` can replay the original
owner's raw float32 latents when its public Task supports them. Never substitute
a smaller clip or different precision for the selected full profile.

The environment needs FFmpeg/ffprobe with `libx264rgb`, the native server and
PyPI AIPerf 0.13.0 plus this repository's qualification plugins. Optional
`video_worker`, `video_port`, and `video_request_timeout` select the server
executable, port and timeout. The opt-in `trtmc_video_capture` transport extends
only AIPerf's actual download hook: normal submission, polling, and download stay
with the published client. It preserves the exact returned bytes by job ID,
because AIPerf deliberately excludes binary downloads from raw JSON exports.
The qualifier checks every decoded RGB frame, frame order, dimensions, file
hash, native completion, fixed request controls and request/metric identity.
It saves complete clips and decoded frames; these can be compared against the
original owning reference with its unchanged quality criteria.

Client latency includes polling, download and capture-file writing. It is
reported separately from native Task time, without subtracting estimated
overhead. Encoding FPS is presentation metadata; it never changes built frames.
The [three-step usage guide](../../website/docs/user-guides/profile-videos-with-aiperf.md)
also works with plain `pip install aiperf==0.13.0`, without an AIPerf checkout
or the optional capture transport.

### Pinned real prompts

Copy `config/text/environment.yaml` and
`config/text/qwen3-0.6b-real-prompts.yaml`, then set their existing interpreter,
server, runtime and bundle paths. Use the already-built `qwen3-0.6b-fp16`
full profile; the runner invokes its unchanged owning-family checks against
that exact bundle before profiling. Build instructions remain in the website's
[Build a Bundle](../../website/docs/user-guides/build-a-bundle.md) guide.

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-real-prompts.yaml \
  --out artifacts/qwen-real-prompts
```

The workload pins the ShareGPT file revision and checksum and the model's
existing tokenizer revision. It selects the first twelve distinct, nonempty first-human
turns whose original text plus sixteen output tokens fits the bundle's recorded
context limit. Selection does not rewrite, truncate or pad text. `corpus.json`
records every considered source index, selected string, tokenizer length and
rejection reason; unexamined rows are counted separately. Counts use the pinned
tokenizer without special tokens; native input-token counts remain unavailable.

The published AIPerf `single_turn` loader sends `inputs.jsonl`, with twelve
measured requests, one separately retained warmup, concurrency one and greedy
sampling. Every selected prompt must appear in the measured phase.
`run-000/corpus-verification.json` joins the original strings to actual wire
payloads and the native-dispatch input/output hashes by request ID.
Server records contain hashes and byte counts, without storing raw text.
`report.json` keeps client latency/throughput and native Task-call timings
separate. Build, loading and warmup are excluded from measured requests.
A selection failure or changed/missing input/output fails verification and
preserves its ledger and request artifacts. Family quality criteria are unchanged;
this workload verifies transport and reports performance, without adding a new
numerical accuracy gate.

### Pinned conversations

Use the same environment and already-built Qwen3-0.6B FP16 bundle with
`config/text/qwen3-0.6b-conversations.yaml`:

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-conversations.yaml \
  --out artifacts/qwen-conversations
```

The current Qwen family library implements the existing native
`TextConversation` contract for supported ChatML templates. The server forwards
typed roles and text; Qwen owns template rendering and tokenization. Existing
single-turn generation and its correctness criteria remain unchanged. Rebuild
the server/runtime libraries for history support; existing engine bundles can
be reused. This profile uses nonstreaming chat; history streaming, tools and
non-text messages are unsupported.

The profile pins the same ShareGPT file and tokenizer as the single-turn
example. It selects six distinct fitting human/assistant/human prefixes and
sends their first two original human strings through AIPerf's existing
`multi_turn` loader. AIPerf captures actual native answers for subsequent
history (`deltas_without_responses`); source assistant answers are not replayed.
The second turn has 50 ms of think time. Twelve measured requests cover both
turns of all six conversations, with two separate warmup requests.

`corpus.json` preserves source indices, original user strings, considered
rejections, the selection budget and the exact delay policy. The run's
`corpus-verification.json` checks roles/order, original text, retained actual
assistant output, native dispatch hashes, and every measured turn by request
and conversation identity. Actual wire histories are counted with the pinned
tokenizer and checked against the bundle context plus requested output;
native input-token counts remain unavailable. Configured and observed
between-turn gaps are recorded separately, without claiming exact scheduler
precision. `report.json` includes client and native Task-call timings for each
turn as well as the full workload. Startup/build/load/warmup are excluded from
measured requests. No TTFT/ITL or cross-request cache acceleration is claimed.

### Pinned production trace

Use the same environment and already-built Qwen3-0.6B FP16 bundle with
`config/text/qwen3-0.6b-trace.yaml`:

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-trace.yaml \
  --out artifacts/qwen-trace
```

The source is the CC-BY [Azure 2023 conversational inference trace](https://github.com/Azure/AzurePublicDataset/blob/790921015d50dd6aae7f7e47f39ba0e235ad6b08/AzureLLMInferenceDataset2023.md)
from Patel et al., *Splitwise* (ISCA 2024). The configuration pins the CSV
revision and checksum. Selection keeps the first twelve source rows with
positive input/output targets whose sum plus a sixteen-token safety margin
fits the existing bundle. `corpus.json` retains original rows, source indices,
selection/rejection reasons and unexamined counts. Targets are never shortened.

Source timestamps have no declared timezone. Preparation subtracts the earliest
selected timestamp using decimal arithmetic, multiplies by `corpus.time_scale`
(default `1`), then writes relative milliseconds for AIPerf's existing
`mooncake_trace` loader and `--fixed-schedule`. There is no concurrency or
request-rate override. Run each time scale separately with a fresh output
directory. Scaling changes offered load; it is always recorded.

This trace contains token lengths, not original prompts. AIPerf synthesizes
input text. The verifier records its actual pinned-tokenizer length and checks
the context bound without truncation. Original output lengths become native
generation caps; EOS can produce fewer tokens, and actual native output counts
remain visible. These token targets do not establish a numerical accuracy test.
The unchanged owning-family checks run on the exact served bundle first.

All twelve arrivals are measured, including the first request. There is no
AIPerf warmup or implicit retry; full validation and server startup precede the
trace window. One replica provides one execution lane. Overload can return
`429`, which is retained as a rejection rather than silently dropped.
`run-000/trace-accounting.json` accounts for completed, rejected and failed
requests by source/request identity, checks input/output hashes and generation
caps, and reports offered/achieved rates. It checks relative scheduler credit
offsets against the recorded `arrival_tolerance_ms` (default 100 ms); actual
HTTP send offsets are reported separately. This tolerance describes replay
timing, without changing model correctness criteria. Startup jitter and client
scheduling can affect send timing; submillisecond replay accuracy is not claimed.

The ordinary AIPerf exports retain client timing and replay metrics.
`report.json` separately reports server admission and `public_task_call_wall`
timing for completed requests. A profile needs native completions, full request
accounting and no unexpected failures. Rejected requests and their client
errors remain visible. Nonstreaming trace replay provides no TTFT/ITL samples.

### Controlled shared prefixes

Use `config/text/qwen3-0.6b-shared-prefix.yaml` with the same environment and
existing full Qwen bundle. Keep exactly one workload case per invocation.
Run each configuration separately so it starts its own server process.
First leave `workload.warmup_requests: 0`:

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-shared-prefix.yaml \
  --out artifacts/qwen-prefix-fresh
```

Copy the configuration, change only `workload.warmup_requests` to `12`, and
run it again with `--out artifacts/qwen-prefix-warm`. Each invocation starts a
new server process and runs unchanged family correctness on its exact bundle.
Both runs measure the same twelve prompts, concurrency one and sixteen-token
generation caps. The second run first sends twelve separate warmup requests
from that same corpus. Startup, validation and warmup are excluded from the
measured request metrics. This is a process-warmup comparison; the fresh run
can still benefit from shared host/system caches.

The layout consists of one literal prefix and twelve distinct suffixes.
`corpus.json` retains each complete original concatenation, UTF-8 hash, actual
pinned-tokenizer length, shared token-prefix length and overlap fraction.
Overlap is counted on complete prompt token IDs, so tokenization at the
prefix/suffix boundary is included. Any over-limit prompt fails selection
with a rejection ledger; prompts are never shortened. AIPerf's existing
`single_turn` loader sends these exact strings. The verification report checks
every measured prompt, native input/output hashes and fresh/warm phase counts.

The qualified Qwen implementation clears generation state in
`reset_generation_context` before each request. It exposes no cross-request
prefix-cache control or cache-hit telemetry. `cache_audit` records this manual
source audit, its source-file hash and repository revision, separately from
observations. Per-generation KV state does not establish reuse across requests.
Cache-hit telemetry is explicitly unavailable. Latency/throughput differences
in `report.json` are descriptive; neither common input text nor a faster warm
run proves a cache hit. No cache implementation or model spec is added.

### Stateless Responses

Use `config/text/qwen3-0.6b-responses.yaml` with the same environment and
already-built full Qwen3-0.6B FP16 bundle:

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-responses.yaml \
  --out artifacts/qwen-responses
```

The existing server accepts nonstreaming `/v1/responses` with one input string
or one user message containing string/`input_text` parts, plus optional string
`instructions`. It maps these onto the same public native generation Task,
admission, limits, errors and timing records as text generation. Model-specific
chat formatting stays in the family. One successful native Task call produces
one typed assistant `message` with `output_text` content, `resp_`/`msg_` IDs and
completed status. Native EOS-versus-cap termination metadata is unavailable.
Input/total token usage is unavailable, so `usage` is null; actual native output
counts remain in the server records. AIPerf uses its default client tokenizer
estimates. Do not request `--use-server-token-count`.

This subset is stateless: `store` defaults to false. Storage, response-ID
continuation, tools, history, media, structured reasoning, and streaming are
explicit errors. No Responses events or TTFT/ITL are advertised. Generation
uses `max_output_tokens` (at least 16) and the existing supported sampling
controls. The reference perf-serving server does not expose this endpoint, so
the optional matched reference-performance comparison is rejected for Responses;
the unchanged family correctness prerequisite still runs.

The pilot selects the first six fitting original prompts from the same pinned
ShareGPT source/tokenizer as the single-turn corpus. Published AIPerf's existing
`responses` endpoint and `single_turn` loader send six measured requests plus one
separate warmup, concurrency one, greedy sampling, a sixteen-token cap and
`store:false`. `corpus-verification.json` checks original text and native hashes;
`responses-verification.json` additionally checks the typed response, model,
unique request/response identities, output, native generation cap and Task timing.
Failures preserve a verdict and the raw/joined request records. `report.json`
keeps client end-to-end metrics separate from native `public_task_call_wall`.

### SpecBench ordinary-generation baseline

Use `config/text/qwen3-0.6b-specbench.yaml` with the same environment and
already-built full Qwen3-0.6B FP16 bundle:

```bash
python -m trtmc_aiperf_qual text-profile \
  --environment environment.yaml --config qwen3-0.6b-specbench.yaml \
  --out artifacts/qwen-specbench
```

The configuration pins [Spec-Bench](https://github.com/hemingkx/Spec-Bench/tree/fd2c1cd7d2201ef71db4c5f4e455008f017967bf),
Xia et al. (ACL 2024 Findings), at revision
`fd2c1cd7d2201ef71db4c5f4e455008f017967bf`, the same source revision used by the
tested AIPerf public loader. The upstream repository carries an Apache-2.0
license. The JSONL checksum is recorded in the configuration. Selection keeps
the first twelve distinct fitting original first-turn strings in source file
order. `corpus.json` records question IDs, categories, original text/hashes,
tokenizer lengths, rejected inputs and unexamined counts. There is no trimming,
rewriting or truncation. This small context-bounded selection does not qualify
the entire source corpus or a SpecBench quality score.

Preparation materializes these exact strings for AIPerf's existing `single_turn`
loader. The public SpecBench loader ordinarily downloads the full corpus and
strips strings; the materialized selection keeps the existing 256-token bundle
limit and original text explicit. No AIPerf source checkout or loader fork is
required. The pilot runs twelve measured completions plus one separate warmup,
concurrency one, greedy sampling and sixteen-token generation caps. Its unchanged
owning-family checks run on the exact served bundle first.
Use one output-token cap per profile; run separate configurations to compare caps
so the recorded and native generation work remain aligned.

This is **ordinary native generation**. The pilot does not configure or qualify
a draft model, speculative orchestration or acceptance counters. `corpus.json`,
the request verification report and `report.json` state `ordinary_native_generation`
and retain a separate manual runtime audit with source-file/revision hashes.
Speculative execution is unqualified and acceptance metrics are unavailable.
The verifier checks exact source/wire/native input and output identity, generation
caps and native Task timing. Client timing and `public_task_call_wall` remain
separate; build/load/startup/warmup are excluded. A corpus name is not evidence
of speculative acceleration. No draft/target model, model spec, or speculative
runtime is added.

### Text workload pilot coverage

These workload configurations use the existing Qwen3-0.6B full FP16 profile.
Hardware qualification evidence covers GB300; other families/targets are not
implied by this pilot table. All clients use the published AIPerf package.

| Workload | Native HTTP endpoint | Qualified workload scope | Capability limit |
| --- | --- | --- | --- |
| Real prompts | `completions` | Twelve original pinned ShareGPT prompts | Selected subset; ordinary generation |
| Conversations | `chat/completions` | Six two-turn conversations, live answers and think time | Nonstreaming native text history |
| Production arrivals | `completions` | Twelve eligible Azure rows; original and scaled arrivals | Length-only source; synthesized prompts; rejections retained |
| Shared prefixes | `completions` | Literal prefix/suffix corpus, fresh/warm process comparison | Cross-request cache-hit telemetry unavailable |
| Responses | `responses` | Six original prompts, stateless typed text responses | No storage, history, tools or streaming |
| SpecBench corpus | `completions` | Twelve fitting original first turns | Ordinary generation only; speculative execution and acceptance metrics unqualified |
