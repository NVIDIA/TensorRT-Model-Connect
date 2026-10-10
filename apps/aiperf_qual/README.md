# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).
It does not use `qualification_tests/benchmark_qualification`.

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
| Model | Derived from the catalog entry and its Task. `config/models/<profile>.yaml` holds only exceptions. | Only exceptions |

`trtmc-aiperf-qual plan` prints the derived configuration of every model; `trtmc-aiperf-qual matrix`
writes the execution matrix (native path, environment, workloads, checks, and whether each profile is
executable).

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

Checks for every model of a Task (`supplementary`): text-to-speech round-trip WER (corpus, bootstrap) and
audio validity (every output finite and not silent; the median per-sentence duration ratio to the native model
within 0.5-2, as a sampling model's single utterance may run to its length limit on either side; an error
when no sentence has audio on both sides); GenEval-style pass rate for text-to-image families that take caller latents (the same
initial noise on both sides); CLIP-T and video validity for videos; MagicBrush CLIP-I and DINO for edits;
world-model video parity. The pixel parity under latent replay is reported only (`informational`).

A suite with `base: catalog` overrides the profile's catalog request with its dataset fields.

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
- Every AIPerf run has a deadline: three times the profile's seconds in the run's ledger (`run-all --ledger`, at
  least ten minutes), else 12 hours; a GPU phase that fails before producing its result runs once more.
- `summary` reports one result per model, worst first: White (no verdict: an error or a failed build; or no valid
  comparison: a Task without an Acc check), Red (Acc worse than
  native beyond its margin), Yellow (an Acc difference not shown either way), Green (quality passes with available
  benchmark timings). Historical fixed-workload reports retain their original performance lights.

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
the configured roots. Reference environments, datasets, runtime binaries and reports are kept.
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
requires generation again; text raw responses remain available. Cleanup does not promise
bit-for-bit model replay without the checkpoint, bundle and generated artifacts.

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


## Text profiling with the persistent server

`text-profile` reuses `trtmc-server` with AIPerf's built-in `completions` and
`chat` endpoints. It adds server readiness/model checks, AIPerf release
version checks and installed-package provenance, family validation commands,
length/load sweeps, request-ID timing joins and an optional sequential
native-reference comparison. It produces
benchmark evidence independently of the qualification verdicts above.

Qwen and GPT-2 prebuilt validation checks the builder-recorded immutable Hugging Face
checkpoint identity and build settings against the selected pinned snapshot and family manifest.
Bundles created before this metadata was added must be rebuilt for automated
exact-bundle validation. Declaring a profile or revision in YAML is insufficient.


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
