---
title: Testing Reference
---

## Documentation validation

`.github/workflows/docs-validation.yml` is the required documentation gate on
every pull request, pushes to `main`, and manual runs. Its `Validate
documentation` job uses Python 3.12 and Node 20 and runs all of these checks:

1. Unit tests for the file-reference, command, runtime-strategy-matrix, and
   selective-impact validators.
2. `tools/test_impact.py --validate` to prove documentation and validator
   changes still select the intended focused checks.
3. Strict tracked-document references and checked numeric claims.
4. Documented shell-command syntax and local CLI/argument contracts.
5. The live runtime-strategy matrix against descriptors, source, tests, and
   runner commands.
6. A clean lockfile install followed by a production Docusaurus build.

Reproduce the full job from the repository root:

```bash
python3 -m pytest \
  tests/tools/test_check_doc_file_references.py \
  tests/tools/test_check_doc_commands.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_owned_validation_scripts.py \
  tests/tools/test_test_impact.py \
  tests/tools/test_trtmc_validate.py \
  tests/tools/test_perf_matrix.py::test_release_suite_covers_every_non_l0_ready_model_profile \
  -q
PYTHONPATH=python:. python3 tools/test_impact.py --validate
python3 tools/check_doc_file_references.py --strict --tracked
python3 tools/check_doc_commands.py
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
npm --prefix website ci
npm --prefix website run build
git diff --check
```

Install `pytest` and `PyYAML` first if the Python environment does not provide
them. `npm ci` uses `website/package-lock.json` and replaces that workspace's
installed dependency tree; use Node 20 to match CI. `git diff --check` is an
additional local patch-quality check rather than a step in the GitHub job.

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

`tools/trtmc_validate.py` is the Dev/QA entry point for proving that one TRTMC
model agrees with its original reference implementation. Model-to-workload
ownership is declared in `tests/validation/model_workloads.yaml`; the command
does not infer a generic task from the model name.

Run a model's default binding, choose another declared workload, inspect the
current all-model plan without executing it, or run every validation-eligible
ready single-device model:

```bash
python3 tools/trtmc_validate.py gpt2-125m
python3 tools/trtmc_validate.py internvl3-8b vlm_mmmu_pro_vision_mcq
python3 tools/trtmc_validate.py --all --dry-run
python3 tools/trtmc_validate.py --all
```

Eligibility excludes manifests that require multiple devices, are marked
`skip`, or use `ci_tier: l0_only`; readiness alone does not select a model. At
this audited repository snapshot, the dry run resolves 105 eligible bindings:
97 use dataset-backed reference workloads and 8 are explicitly marked
`not_compared_reason` because no independent comparator is currently
available. Treat those numbers as a repository snapshot, not a permanent API
promise; rerun `--all --dry-run` after catalog changes. Every runnable binding
must select an independent native reference runner and cannot silently fall
back to either model-owned E2E or the `tools/task_eval.py` CLI.

The all-model supervisor uses one isolated worker process per model and, by
default, records a failed worker before continuing. Use
`--all --on-model-failure stop` to stop after the first failed model. Both
policies return nonzero when any attempted model fails.

Each case writes
`<output>/<model>/<workload>/comparison.json`; the output root receives
`report.json` and the **TRTMC Reference Consistency Report** in `report.html`.
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
`--dry-run` proves binding and planning only; documentation CI does not execute
the 97 runnable comparisons in the 105-entry plan. See
`tests/validation/README.md` for the artifact and model-onboarding contracts.

## Model-owned E2E

Replace placeholders with literal values:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/<family> \
  --e2e-model <manifest-name> \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

Add `--hf-python /path/to/python` only when the selected runtime needs a Python
helper. E2E requires the checkpoint, a compatible GPU/TensorRT environment,
and the CLI binary. The remaining runtime evidence depends on the bundle path:
a native bundle needs its owning model and backend DSOs; an optimized-runtime
bundle must contain the qualified implementation metadata, integrity-bound
artifact tree, and embedded implementation DSO.

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
- an exact qualified profile under `profiles/*.toml`, binding the model
  revision, target, options, and semantic hash
- a matching model-owned `QUALIFICATION.*.toml` producer descriptor and its
  retained parity/performance artifacts
- a built bundle containing `optimized_runtime.json`, `implementation.json`,
  the integrity-bound artifact tree, and the embedded implementation DSO

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
| Build/selection identity | Family `MODEL.toml` and exact `runtime_strategy` | `IMPLEMENTATION.toml` plus an exact qualified `profiles/*.toml` entry |
| Qualification authority | Model-owned E2E manifest and retained comparison artifacts | Matching `QUALIFICATION.*.toml` producer proof and retained parity/performance artifacts |
| Bundle dispatch | `config.json` and `runtime_strategy` | `optimized_runtime.json` and `implementation.json` |
| Runtime libraries | Owning `libtrtmc_model_*.so` and selected `libtrtmc_backend_*.so` | Exact embedded `libtrtmc_impl_*.so`; no native strategy/model/backend dispatch |
| Timing evidence | Provider-populated `setup_ms`, `prefill_ms`, and `decode_ms`, when available | Provider-populated phase timing when available; otherwise synchronized public-call wall time |

The CLI prints the phase fields returned by `TextResult`, but providers are not
required to expose every phase. A zero phase value can mean unavailable; for
example, the qualified Qwen Edge-LLM adapter deliberately reports zero
prefill/decode timing because its pinned downstream API has no trustworthy
split. Do not turn those zeros into latency or throughput claims. Use the
qualification performance runner or another synchronized wall-clock
measurement and label that metric explicitly.

## Choosing evidence

| Change | Minimum useful evidence |
| --- | --- |
| Python family plugin | Focused builder tests and one representative E2E case |
| Native runtime model DSO | Focused C++ tests, strategy/descriptor checks, backend-load evidence, and matching E2E |
| Optimized implementation | Implementation/profile/qualification contract tests, embedded-DSO host tests, and matching qualified E2E/performance artifacts |
| Shared runtime/config | Focused unit tests plus affected-model selection |
| Public C++ API / C-linkage subset | API/ABI tests and CLI smoke |
| E2E runner/comparator | Focused harness tests and representative artifact |
| Quantization | Builder checks plus model/modality parity and health evidence |
| Documentation commands | Parser/help check, path check, and execution where dependencies permit |

Compilation is not parity. A single-model E2E is not broad regression proof.
Performance results are not qualification evidence unless the run also records
the exact commit, hardware, inputs, artifacts, and comparison baseline.
