---
title: Testing Reference
---

## CI and evidence boundary

Source contains the test implementation and four GitHub workflows: the public
CPU request broker and executor, the Internal CI Bridge, and the path-scoped
Pages deployment. Protected premerge, including legal compliance, and nightly
orchestration run in private Internal CI.

The pull-request author first comments `/run-ci` to run public CPU validation
on the exact current PR merge. After the public required check passes, an
authorized collaborator with maintain or admin access applies
`run-internal-ci` to dispatch the exact current PR head. Source receives
contributor-visible public CPU checks and only the sanitized
`trtmc/premerge/required` status from protected CI. Raw protected logs,
artifacts, runner details, internal packages, and the complete report remain
private. Neither path triggers on a push to `main`, so merging a passing PR
does not rerun the same premerge suite.

Source has no separate pull-request documentation-validation workflow. The
Pages workflow builds the site before deployment from `main`, but it is not a
pull-request documentation gate.

## Local documentation validation

Use the checks that are present on this snapshot. The following is a
recommended local set, not one pre-existing required workflow:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_check_doc_file_references.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_owned_validation_scripts.py \
  tests/tools/test_validation_engine.py \
  tests/tools/test_test_impact.py \
  tests/tools/test_trtmc_validate.py \
  tests/tools/test_perf_matrix.py::test_release_suite_covers_every_non_l0_ready_model_profile \
  -q
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
python3 tools/check_doc_file_references.py --strict website/docs
npm --prefix website ci
npm --prefix website run build
git diff --check
```

Install `numpy`, `Pillow`, `pytest`, and `PyYAML` first if the Python
environment does not provide them. `npm ci` uses
`website/package-lock.json` and replaces that workspace's installed dependency
tree; use Node 20 to match `.github/workflows/pages.yml`. The reference checker
validates repository-relative paths and selected numeric claims. The
runtime-strategy checker compares native descriptors, source, tests, and runner
commands. A successful Docusaurus build proves that this site is buildable; it
does not by itself prove every documented behavior or command.

The runtime-strategy checker is a useful drift diagnostic, but it is not green
on this snapshot:

```bash
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
```

At commit `e6b798cdb145c38caf1ede8eda7f5ce83f894138`, it exits nonzero because
`diffusion_sana_wm` is absent from both
`tests/runtime_strategy_matrix.yaml` and the E2E manifests. It also finds no
discoverable runner class for Canary and Whisper speech-to-text, Nemotron
streaming speech-to-text, PersonaPlex speech-to-speech, or Qwen3-Omni
omni-multimodal. Treat these as codebase validation gaps; a documentation-only
change must not claim that this checker passes.

## Active workflow inventory

Source contains exactly these four workflow files:

| Workflow | Trigger and evidence boundary |
| --- | --- |
| `.github/workflows/community-cpu-request.yml` | Exact `/run-ci` PR comment from the PR author or a maintainer/admin; dispatches only an exact public PR snapshot without executing PR code. |
| `.github/workflows/community-cpu.yml` | Manual dispatch from the trusted request broker; runs read-only public CPU validation and publishes checks on the exact PR merge SHA. |
| `.github/workflows/internal-ci-bridge.yml` | One-shot `run-internal-ci` label or manual request; authorizes the actor, verifies current public CPU success, captures the exact PR head, and dispatches private premerge. |
| `.github/workflows/pages.yml` | Pushes affecting `website/**` on `main`, or manual runs; builds and deploys only the documentation site to GitHub Pages. |

Internal scheduled nightly and model proof are not Source workflows. Their raw
evidence is not published through Source Actions or Pages. Source currently has
no active optimized-runtime hardware qualification route.

## CPU and repository checks

Run Python tests with both the package and repository root importable:

```bash
PYTHONPATH=python:. python3 -m pytest tests/builder -q
PYTHONPATH=python:. python3 -m pytest tests/tools -q
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
```

Run compiled tests after configuring and building the project:

```bash
ctest --test-dir build --output-on-failure
```

Some C++ tests require TensorRT, CUDA, a GPU, or model plugins; configure the
build in the supported development environment and inspect CTest labels before
assuming the whole suite is CPU-only.

## Model-first reference consistency

`tools/trtmc_validate.py` is the supported Dev/QA entry point for proving that
one TRTMC model agrees with its original reference implementation. The shared
execution implementation now lives in `tools/validation/engine.py`; the former
task-eval executable has been retired. Prepared manifests retain the `task_eval`
metadata key only as a stable artifact and reference-cache schema.
Model-to-workload ownership is declared in
`tests/validation/model_workloads.yaml`; the command does not infer a generic
task from the model name.

Run a model's default binding, choose another declared workload, inspect the
current all-model plan without executing it, or run every validation-eligible
ready single-device model:

```bash
python3 tools/trtmc_validate.py gpt2-125m
python3 tools/trtmc_validate.py internvl3-8b vlm_mmmu_pro_vision_mcq
python3 tools/trtmc_validate.py --all --dry-run
python3 tools/trtmc_validate.py --all
```

Qwen3-Omni now has a model-plugin validation binding, and the validation
runner forwards `--model-plugin-dir` to each Omni runtime stage:

```bash
python3 tools/trtmc_validate.py \
  qwen3-omni-30b-a3b-instruct \
  seedtts_en_omni_audio_parity \
  --model-plugin-dir ./build/models
```

That workload reads
`/mnt/data/seedtts-en-omni-audio/dataset.json` and is declared `local_only` for
a GB300 environment with roughly 280 GB of model/reference state. It is not a
host-only documentation check; run it only in the prepared target environment
and retain its model artifacts and comparison report.

Eligibility excludes manifests that require multiple devices, are marked
`skip`, or use `ci_tier: l0_only`; readiness alone does not select a model. At
commit `e6b798cdb145c38caf1ede8eda7f5ce83f894138`, an actual
`--all --dry-run` resolves 105 bindings, all runnable; the current catalog has
zero `not_compared` entries. Treat those counts as snapshot evidence, not a
permanent API promise; rerun the dry run after catalog changes. Runnable
bindings use the reference runner selected for their prepared dataset kind. If
a future model lacks an implemented aligned comparison, it remains visible as
`not_compared` instead of launching a model worker.

The supervisor starts one isolated worker process per **attempt**. By default,
each runnable binding permits at most two attempts and waits five seconds
before the second attempt. Only an execution error is retried; a completed
comparison disagreement is final and is never retried. Configure those limits
with `--model-attempts` and `--model-retry-delay-seconds`. Intermediate
execution-error artifacts are archived, and `comparison.json` records the
attempt count and per-attempt evidence.

After a binding reaches its terminal result, the all-model failure policy
applies. By default the supervisor records a failed binding and continues with
the remaining models. Use `--all --on-model-failure stop` to stop after the
first terminally failed model. Both policies return nonzero when any attempted
model ultimately fails.

Each case writes
`<output>/<model>/<workload>/comparison.json`; the output root receives
`report.json` and the **TRTMC Reference Consistency Report** in `report.html`.
The run also freezes its ordered selected-case inventory in
`<output>/ledger/campaign.json` and atomically maintains one receipt per case
under `<output>/ledger/cases/`. `report.json` is rebuilt from those receipts,
so it includes pending, running, and terminal cases while the command is still
running. On `--resume-existing`, a leftover running attempt becomes
interrupted, completed receipts remain unchanged, and only unfinished cases
can start a new case attempt. Model-worker retries remain nested evidence
within that case attempt; they are not additional ledger attempts.
`comparison.json` remains an atomically repaired compatibility projection.
A not-compared entry writes
`<output>/<model>/not-compared/comparison.json` without launching a worker.
Exit status `0` means all attempted comparisons passed, `1` means execution
completed but validation failed, and `2` means CLI/setup validation failed or
a single requested model is explicitly not compared. During `--all`,
not-compared entries make the aggregate report incomplete but do not by
themselves fail the process. The report keeps execution, comparison, and final
validation status separate and records bounded reproduction evidence.

Dataset-backed workloads use the sample limit declared for their task. Override
one run with `--limit`, where zero requests the complete dataset:

```bash
python3 tools/trtmc_validate.py gpt2-125m --limit 100
python3 tools/trtmc_validate.py gpt2-125m --limit 0
```

`--limit` applies only to runnable dataset-backed workloads. Not-compared
entries have no dataset slice, report `sample_limit: 0` in a dry-run plan, and
do not launch a worker.

This workflow needs the model checkpoint, its reference environment and
dataset, a compatible TRTMC bundle/runtime, and usually target GPU hardware.
`--dry-run` proves binding and planning only; it does not execute the 105
runnable comparisons in the current 105-entry plan. See
`tests/validation/README.md` for the artifact and model-onboarding contracts.

## Model-owned E2E

This concrete command runs the current Qwen native L0 manifest. Replace its
literal family and manifest values when validating another owner:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/qwen \
  --e2e-model qwen3-0.6b-native-l0 \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

Add `--hf-python /path/to/python` only when the selected runtime needs a Python
helper. E2E requires the checkpoint, a compatible GPU/TensorRT environment,
and the CLI binary. The remaining runtime evidence depends on the bundle path:
a native bundle needs its owning model and backend DSOs; an optimized-runtime
bundle must contain its declared implementation metadata, integrity-bound
artifact tree, and embedded implementation DSO. Target-environment parity and
performance proof is a separate evidence layer.

`tests/test_e2e.py` is the repository-wide compatibility entry point. Model
work should normally select the owning `tests/e2e/models/<family>/` tree so
collection, defaults, waives, artifacts, and impact remain model-local.

## Native manifest and optimized-provider contracts

Each `tests/e2e/models/<family>/MODEL.toml` declares that family's JSON
manifests. Each buildable **native** JSON manifest requires:

- `name`, `hf_id`, `bundle`, and `family`
- an exact family-owned `runtime_strategy`
- a `task_strategy` or a runtime-strategy mapping from
  `tests/runtime_strategy_matrix.yaml`
- a non-empty `testcases` array

Each testcase carries the request and oracle contract, including fields such
as `name`, `user_contract`, `ci_tier`, prompt/media inputs, reference-family
metadata, and thresholds. Fields shared by every testcase stay at manifest
level; testcase values override inherited defaults.

An optimized implementation has a separate evidence chain:

- family-owned `IMPLEMENTATION.toml`, including the private
  `libtrtmc_impl_*.so` factory identity
- an exact profile under `profiles/*.toml`, binding the model revision, target,
  options, declared qualification state, and semantic hash
- focused implementation/profile and bundle-contract tests
- a built bundle containing `optimized_runtime.json`, `implementation.json`,
  the integrity-bound artifact tree, and the embedded implementation DSO

Source currently has no active optimized-runtime hardware qualification route.
A profile's qualification-state field participates in declaration and routing;
it is not, by itself, current Source hardware proof. Target-environment or
private parity/performance evidence must be evaluated separately.

`PipelineFactory` checks `optimized_runtime.json` before native strategy
dispatch. Consequently, an optimized bundle's public `runtime_strategy` may be
empty: its implementation ID and profile ID are the selection evidence, and it
does not load a native model or backend DSO through the strategy registry.

Do not copy a generic example and invent either a runtime strategy or an
optimized profile. Use the matching Python, runtime, and E2E descriptors as the
source of truth.

## Runtime evidence by bundle path

| Evidence | Native bundle | Optimized-runtime bundle |
| --- | --- | --- |
| Build/selection identity | Family `MODEL.toml` and exact `runtime_strategy` | `IMPLEMENTATION.toml` plus an exact `profiles/*.toml` entry |
| Qualification authority | Model-owned E2E manifest and retained comparison artifacts | Target-environment/private proof outside Source; Source publishes no active optimized-runtime hardware qualification route or raw proof artifacts |
| Bundle dispatch | `config.json` and `runtime_strategy` | `optimized_runtime.json` and `implementation.json` |
| Runtime libraries | Owning `libtrtmc_model_*.so` and selected `libtrtmc_backend_*.so` | Exact embedded `libtrtmc_impl_*.so`; no native strategy/model/backend dispatch |
| Timing evidence | Provider-populated `setup_ms`, `prefill_ms`, and `decode_ms`, when available | Provider-populated phase timing when available; otherwise synchronized public-call wall time |

The CLI prints the phase fields returned by `TextResult`, but providers are not
required to expose every phase. A zero phase value can mean unavailable; do not
turn zeros into latency or throughput claims. If synchronized public-call wall
time is used instead, label the metric and measurement boundary explicitly.

## Choosing evidence

| Change | Minimum useful evidence |
| --- | --- |
| Python family plugin | Focused builder tests and one representative E2E case |
| Native runtime model DSO | Focused C++ tests, strategy/descriptor checks, backend-load evidence, and matching E2E |
| Optimized implementation | Source implementation/profile/bundle contract tests and embedded-DSO host tests, plus target-environment parity/performance proof when required; Source CI does not supply that hardware proof |
| Shared runtime/config | Focused unit tests plus affected-model selection |
| Public C++ API / C-linkage subset | API/ABI tests and CLI smoke |
| E2E runner/comparator | Focused harness tests and representative artifact |
| Quantization | Builder checks plus model/modality parity and health evidence |
| Documentation commands | Parser/help check, path check, and execution where dependencies permit |

Compilation is not parity. A single-model E2E is not broad regression proof.
Performance results are not qualification evidence unless the run also records
the exact commit, hardware, inputs, artifacts, and comparison baseline.

{/* Collaborative review anchor: batch 2. */}
