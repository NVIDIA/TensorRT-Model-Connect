<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy and Performance qualification design

Status: the draft now implements common/family environment selection, preparation,
GPT-2 compiled Performance, stability checks, optional run targets, and report
evidence. See README.md for the executable interface. Historical benchmark
migration beyond GPT-2 continuation parity and target-hardware qualification
remain separate acceptance work; this document is not a record of passing GPU runs.

## Historical basis

The comparison baseline is commit
`3d7f5147811a7a7ac8a441cee7aa2042e8d8fc34`, the first parent of PR #1093's
merge commit `c19a3e6d8d112cd1b2c3d02f6c39c408ac8e83e8`.

Relevant historical sources:

- `tests/e2e/models/gpt2/e2e_plugins/references/hf_transformers.py` and the
  family comparators: independent HF execution and conversion comparisons.
- `tests/e2e/models/gpt2/thresholds/gpt2-125m.json`: model-specific numerical
  and output criteria, not a universal exact-token rule.
- `scripts/eval_mmlu.py`: MMLU answer accuracy. This is distinct from comparing
  HF and TensorRT continuation tokens on MMLU prompts.
- `benchmarks/performance/README.md`, `release.yaml`, and
  `tools/perf_matrix.py`: paired measurements, comparable timing boundaries,
  output validation, stability checks, preparation evidence, and reports.
- `python/tensorrt_model_connect/python_profiles.toml`: a shared base and
  separate dependency environments existed, but used centralized registration.

Preserve the useful evaluation behavior. Keep #1093's family ownership and
public build/bundle/Task boundaries. Do not restore the old central model
catalog, profile registry, E2E plugin system, or imports from retired harnesses.

## Ownership and files

```text
apps/benchmark/
  qualification/
    suites/                         # reusable benchmark definitions
    environments/                   # machine paths, common Python, storage
    runs/                           # selected work, environment, performance targets
  trtmc_benchmark/                   # discovery, processes, generic metrics, reports
families/<family>/
  requirements.txt                  # existing optional family dependencies
  tests/qualification/
    <model>.accuracy.yaml           # suites and cases for this model
    <model>.performance.yaml
    executor.py                    # family comparison and reference behavior
    prepare_environment.py         # optional family environment preparation
    data/                          # optional small committed fixtures
```

Discover only the explicit Accuracy/Performance files in this directory.
Existing E2E and L0 manifests do not opt in. Each file can contain multiple
suites and cases. Keep exact-name model/suite/case selection. Adding a model
to an existing benchmark requires changes only in its family directory.

Shared suite definitions describe datasets, sampling rules, metric definitions,
and measurement protocols. They contain no model list, Python paths, or GPU
selection. Models reference a shared definition by its descriptive filename;
they do not copy that definition. A genuinely new benchmark may add a shared
definition once. Do not add version aliases or a configuration inheritance system.

The family owns checkpoint selection, input preparation, HF or official reference
calls, output interpretation, comparison tolerances, and model-specific validation.
The application can share JSON I/O, subprocess management, generic statistical
functions, and rendering. It cannot dispatch model implementations by family name
or force every model to use a shared Transformers adapter.

The executor is one family entry point, not a new model runtime. It consumes
the existing public build and benchmark interfaces. Split family-local helpers
only when this improves readability; do not require a fixed collection of files
for each new family. Small model-independent scoring functions may be shared;
tokenization, generation, and model-specific comparison decisions remain local.

## Python environments

Use the common Python environment by default. A family that needs incompatible
dependencies supplies the optional `prepare_environment.py` in its own test
directory. Discovery finds that file automatically; there is no central
family-to-environment mapping and no environment field in each model YAML.

The high-level machine configuration supplies only infrastructure, for example:

```yaml
tools:
  python: /opt/trtmc-common/bin/python
storage:
  environment_root: /data/trtmc-python
```

These are proposed fields. `tools.python` is the default execution interpreter,
not necessarily the interpreter running the scheduler. The scheduler reads
configuration and starts processes without importing family modules, Torch,
Transformers, or model builders.

Environment selection has three cases:

| Family needs | Executor and build | HF / official reference |
| --- | --- | --- |
| Common environment works | Common Python | Same Python |
| Family needs other dependencies | Family Python | Same family Python |
| Reference conflicts with build dependencies | Common or family Python | Separate family reference Python |

Use one optional preparation entry point with the same file-based subprocess
convention as the executor: `--request <json> --output <json>`. It runs under the
common Python using standard-library bootstrap code. Its request contains the
common interpreter, family source directory, selected cases, a writable
family environment directory, and whether creation is allowed. It returns:

```json
{
  "python": "/data/trtmc-python/example/build/bin/python",
  "reference_python": "/data/trtmc-python/example/reference/bin/python"
}
```

`reference_python` is optional and defaults to `python`. With no preparation
entry point, both default to the common interpreter. Most families need neither
an extra file nor an extra environment. Preparation can return an already
provisioned environment after checking it; it need not recreate one for each run.

The family owns its preparation recipe and reference-specific installation
steps. Reuse the existing root `requirements.txt` for family build dependencies;
do not add nested requirements files or revive `python_profiles.toml`. A
reference-only environment need not install the TRTMC builder package: its
`transformers==5.2.0` dependency must not force the reference version.
If a family needs another Python version, its preparation recipe resolves an
available interpreter or provisions it during the explicit preparation stage.

Preparation must never install into or upgrade the common environment. A private
environment must have its own dependency resolution; do not overlay incompatible
packages through `PYTHONPATH` or silently inherit system site packages. Check
actual imports and required capabilities, including CUDA and compilation for
Performance, before accepting an environment. Package metadata alone is not a
sufficient compatibility check. Failures are recorded against the owning family;
other independent families remain runnable. Do not retry failed model evaluation
in another environment or downgrade compiled execution to eager execution.

Preserve the selected interpreter path, including virtual-environment symlinks.
Launch Python benchmark/build entry points explicitly with that interpreter,
such as `python -m trtmc_benchmark`, so a globally installed command's shebang
cannot send the builder back to the common environment. The native worker keeps
its existing public interface. Reference subprocesses use the resolved reference
interpreter. All processes receive the high-level device allocation.

No general environment registry, dependency solver service, or cross-family
environment deduplication is needed. Prepare into a fresh directory and publish
it only after validation; reuse a prepared environment without mutating it while
tests are running. Record the resolved interpreter paths, actual dependency
versions, and preparation commands in the run's environment receipt.

## Execution workflow

```text
discover configurations and select cases
  -> prepare/check each selected family's Python environment
  -> prepare and identify bundles and reference checkpoints
  -> execute Accuracy or paired Performance cases
  -> collect family results and produce JSON/HTML reports
```

Retain `plan`, `run`, `resume`, and `report`. Add `prepare` to resolve environments
and bundles without measuring. `run` may perform preparation first for local
convenience; controlled CI runs consume prepared receipts with creation disabled.
`plan` remains read-only: no package installation, model import, or GPU work.
Preparation runs once for a family's selected work, not once per sample.

Separate source/configuration selection from the resolved execution receipt.
Store source revision, resolved configurations, checkpoint revisions, bundle
identity/build provenance, environment evidence, and actual commands. A path
alone is not proof that a bundle belongs to the current conversion. Use existing
build evidence where available; otherwise build a fresh run-owned bundle and
retain its build command and receipt. Do not invent a second builder cache.

Resume must verify these inputs before reusing completed results. An interpreter
path with changed packages is not the same environment. Archive attempts rather
than mixing new measurements into old results. Installing dependencies, building
engines, and downloading checkpoints all finish before Performance timing starts.

The high-level run owns device allocation, exact model selection, concurrency,
and any device-specific Performance target. Families contain no device inclusion
or exclusion rules. Default to serial paired measurements on each allocated GPU;
an outer CI scheduler can run independent families on different allocations.
Use the same CLI for local and CI execution.

## Accuracy behavior

Every conversion case runs the original reference and the converted bundle on
the same resolved input samples. Record checkpoint revision, tokenizer/input
policy, precision on each side, decoding settings, and actual sample identities.

Keep distinct benchmark meanings distinct:

- `mmlu_continuation_parity` compares continuations on the chosen MMLU prompts.
  It is not MMLU answer accuracy.
- A separate MMLU answer-accuracy definition scores both HF and TensorRT against
  the same answer key and reports both scores, their difference, and disagreements.
- Other historical benchmarks retain their own datasets, scorers, numerical
  comparisons, and accepted tolerances in the owning families.

An HF-only score or a successful TRT call does not complete conversion validation.
Do not impose GPT-2 exact-token equality on image, audio, or other language-model
contracts. Do not substitute the current ten-sample GPT-2 smoke case for the old
suite coverage. Migrate historical benchmarks individually with their original
sample selection, prompts, comparator behavior, and thresholds documented.
Unmigrated benchmarks remain explicitly unimplemented, not implicitly passing.

## Performance behavior

For GPT-2 and HF text-generation comparisons, the required baseline is HF
`torch.compile(model.forward)`. Eager results may be diagnostic but cannot replace
this baseline. Other families own the equivalent compiled reference operation;
unsupported compilation is an explicit incomplete comparison.

Both sides use matched workloads, precision policy, output lengths, device
allocation, and measurement boundaries. Record whether tokenization, input
preparation, transfers, and postprocessing are timed. A boundary mismatch or
output-contract failure produces no speed classification. Compilation must be
applied to the callable actually measured, with successful warmup and evidence
of the selected mode; merely recording the requested mode is insufficient.

After output validation, report both raw sample sets, p50/p95, throughput, and
`reference_over_candidate_p50`. Model loading, engine building, compilation,
warmup, and report telemetry are excluded from timed samples. Validate actual
GPU identity and synchronization, and run the two sides sequentially.

Restore the historical stability protocol explicitly: ten samples per side;
first-five and last-five medians differ by at most 5%, and at least eight samples
are within 5% of that side's median. Retry both measurements once in fresh
processes if unstable. Persistent instability is `measurement_inconclusive`,
not a speed regression or a successful comparison. A case using twenty samples
needs an explicitly defined alternative protocol; do not silently reinterpret
the historical rule.

Generic timing/stability calculations can be shared. Output equivalence remains
family-owned; move GPT-2's `exact_token_ids` choice out of the generic text
Performance definition. A high-level run may specify a speed target and whether
it blocks CI. Without a target, a valid ratio remains observation-only. An
execution error still fails operational coverage even for an observation-only
case. Do not equate the old faster/equivalent/slower classification with the
independent decision to fail CI.

## Reports

Keep one authoritative `report.json` and an HTML rendering of that same data.
The common envelope contains case identity, execution status, comparison
validity, metrics, gate evaluations, and artifact links. Family-specific details
can remain under `details`; the renderer must not recognize model names.

Expose selected, completed, comparable, failed, errored, inconclusive, and
observation-only counts without treating them as disjoint categories. Show
operational coverage separately from the number of comparisons meeting a gate.
Errors and unstable measurements remain visible even when no valid speed ratio
exists. An empty discovery plan is empty, not evidence of benchmark coverage.

Accuracy rows show both scores or comparison metrics, sample counts, thresholds,
and disagreements. Performance rows show both backends, both precisions,
compilation/timing evidence, stability, raw samples, ratios, and optional targets.
Both link to the actual leaf commands, logs, environment and bundle receipts,
and reference checkpoint revision. Keep large datasets and weights external;
store report artifacts under the run output and publish through CI artifacts,
not GitHub Pages.

## Changes to the current draft

1. Keep discovery, family YAMLs, shared suite references, file-based executors,
   and JSON/HTML reporting. Add optional family environment preparation and
   propagate the selected interpreter through executor, build, and reference.
   Replace the global reference-only Python setting with the resolved per-family
   receipt. Test common reuse, family isolation, split reference dependencies,
   and interpreter propagation through the real subprocess launch paths.
2. Remove the GPT-2 qualification dependency on a machine-configured
   `hf_transformers_runner` path. Move its reference behavior into the GPT-2 test
   directory, retaining reviewed timing behavior. Leave legacy benchmark tools
   available to their existing consumers; qualification must not import their
   catalog or execution internals. Verify reference code is available in both
   source-checkout and packaged execution.
3. Restore Performance stability and comparable-result reporting; add shared
   report fields for environment and conversion evidence. Verify malformed or
   missing reference results, output mismatches, wrong timing/compile modes,
   instability, stale receipts, and family preparation failures cannot produce
   a successful comparison. Do not weaken existing Accuracy thresholds.
4. Validate GPT-2 Accuracy and compiled HF-vs-TRT Performance on GB300-1, then
   validate an existing family whose reference dependencies conflict with the
   common environment. Prove its addition touches only that family's test files
   and that the common environment remains unchanged. Migrate remaining old
   benchmark contracts family by family with explicit coverage accounting.

CPU tests exercise distinct executor/reference environments and stability/error
handling. They do not prove full historical benchmark coverage, compatibility of
every family, or target-hardware performance. Those acceptance conditions require
their own retained evidence.
