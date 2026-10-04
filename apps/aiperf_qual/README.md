# trtmc-aiperf-qual

Accuracy and performance qualification of every ready catalog model against its native
(unconverted) Hugging Face / PyTorch model, driven by [AIPerf](https://github.com/ai-dynamo/aiperf).
The scheme, its statistics, and its per-Task contracts are in [DESIGN.md](DESIGN.md); it does not use
`qualification_tests/benchmark_qualification`.

- **Acc**: TRTMC must be as accurate as the native model on the Task's workloads. Both sides answer the
  Task's gold-labelled benchmarks (`absolute`) and each benchmark is a paired non-inferiority decision
  (`noninferiority.py`): `pass` when TRTMC's regression is shown to be below the benchmark's margin, `fail`
  when it is shown to exceed it, `inconclusive` otherwise. Tasks without a gold set compare outputs with
  the native model (conversion parity). Random-weight test models are Perf only (`accuracy_source: none`).
- **Perf**: TRTMC must be faster than the native model (eager) at the candidate's precision: the speedup's
  90% interval lies above 1.05, on every timed request, with the same work on both sides.

## Design

| Layer | What | Model-specific? |
|---|---|---|
| Execution | Two HTTP servers speaking the same `/v1/tasks/{operation}` protocol: TRTMC (`trtmc-perf-serve --backend trtmc`, later `trtmc-server`) and the native reference (`--backend reference`: the generic adapter of the operation, or a family's own pipeline, `families/<family>/tests/native_reference.py`). AIPerf sends every request. | No (adapters: per family) |
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
Each benchmark's `margin`, `relative_margin`, `min_native` (suitability floor), and sizes are argued in
DESIGN.md Section 4.

Conversion-parity benchmarks (`gold_metrics.PARITY`) compare each output with the native one within a
tolerance: raw encoders' vectors, MoGe geometry, ACT action chunks, stereo disparities, PersonaPlex speech.
Inputs a family prepares itself come from its `native_inputs.py` (`family_inputs` suites).

Checks for every model of a Task (`supplementary`): text-to-speech round-trip WER (corpus, bootstrap) and
audio validity; GenEval-style pass rate for text-to-image families that take caller latents (the same
initial noise on both sides); CLIP-T and video validity for videos; MagicBrush CLIP-I and DINO for edits;
world-model video parity. The pixel parity under latent replay is reported only (`informational`).

A suite with `base: catalog` overrides the profile's catalog request with its dataset fields.

### Performance

**L1** times each timed request on both servers (warmup, then N requests, R runs): the catalog testcase
(its greedy variant when it samples text; the first gold problem when the testcase is no workload) and, for
text generation, a near-capacity request filling the bundle. The statistic is the server-side model-call
p50 per run; the speedup interval is Welch's t interval of the log ratio. A comparison is white when the
work or outputs differ (checked on every timed response: tokens or text, media geometry, audio length), a
side's runs spread more than 5%, the native model ran at another precision than the candidate, or the GPU was
busy. Timing phases hold the host GPU lock
(`gpu_lock`), which bundle builds on the same host also take. `torch.compile` (`reference_modes`) and the
**L2** serving sweeps (`performance.l2`) are opt-in reports outside the verdict.

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
trtmc-aiperf-qual summary gb300-1=nvidia@host1:/runs/out gb300-2=nvidia@host2:/runs/out \
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
- Model whose native pipeline the generic adapters cannot run: `families/<family>/tests/native_reference.py`
  (an `Adapter(spec)` with `invoke(request, artifact_base)`), named by `reference.adapter`.
- Model needing a differently built bundle: `candidate.build` (or `candidate.model_directory`) in
  `config/models/<profile>.yaml`; the report names the bundle it qualified.
- New machine: a new file under `config/environments/` (paths, Python interpreters, ports, lock, model list,
  retention); Docker or bare metal only differ in these paths.
