<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TRTMC reference validation

`trtmc-validate` is an internal Dev/QA workflow for checking that a TRTMC model
still agrees with its original reference implementation.

Run every Accuracy benchmark configured for a model:

```bash
python tools/trtmc_validate.py gpt2-125m
```

Run one benchmark explicitly:

```bash
python tools/trtmc_validate.py qwen25vl-3b vlm_mmmu_pro_vision_mcq
```

Select multiple models with the same model-first interface:

```bash
python tools/trtmc_validate.py \
  --model gpt2-125m \
  --model qwen25vl-3b
```

Select one or more exact workloads for the selected model set:

```bash
python tools/trtmc_validate.py \
  --model qwen25vl-3b \
  --workload vlm_mmmu_pro_vision_mcq \
  --workload vlm_mmmu_pro_vision_fixed_mcq
```

Automation that already resolved a heterogeneous set of model/workload pairs
can pass exact bindings without constructing a shared workload intersection:

```bash
python tools/trtmc_validate.py \
  --binding qwen25vl-3b=vlm_mmmu_pro_vision_mcq \
  --binding gpt2-125m=mmlu_continuation_parity
```

`--model-selection FILE` accepts the owner/family JSON emitted by
`tools/model_ci.py` and expands it to matching ready model profiles. Every
resolved `(model profile, workload)` pair is an independent result binding.

Run every single-device model whose catalog status is `ready`:

```bash
python tools/trtmc_validate.py --all
```

`--all` expands every `qualification` binding in the model owners'
`validation.yaml` files. There is no separate default, additional, or
diagnostic workload category. A suite that exists in `workloads.yaml` but is
not qualified by a model owner is omitted from model and all-model runs.

The model-owned selection schema is intentionally direct:

```yaml
bindings:
  benchmark_a:
    models: [model-a]
    qualification:
      model-a: {}
  benchmark_b:
    models: [model-a]
    qualification:
      model-a: {}
```

Selecting `model-a` runs both benchmarks as independent bindings.

Configured sample counts use the shared suite's `sample_limit` in
`workloads.yaml`. A heterogeneous full run resolves each binding's own sample
count independently. Do not pass `--limit` for that path: it is a
one-off override applied uniformly to every selected binding, with `--limit 0`
retained as a compatibility spelling and `--limit -1` meaning the complete
datasets. A positive limit larger than a dataset uses every available sample.
Audit the resolved scope and configured sample counts with:

```bash
python tools/trtmc_validate.py --list
```

The all-model command supervises one isolated worker process per model. By
default it records a failed worker and continues with the remaining models.
Stop after the first failed model when that is preferable:

```bash
python tools/trtmc_validate.py --all --on-model-failure stop
```

Both policies return a nonzero exit status when any attempted model fails.
Process isolation also covers failures that happen before backend execution,
such as reference-environment setup or uncaught model-specific Python errors.

Resume an interrupted output only from the same source revision and the same
resolved command. Terminal execution results, including disagreements and
exhausted worker errors, are kept; incomplete or malformed bindings run again:

```bash
python tools/trtmc_validate.py --all \
  --output /runs/results/accuracy \
  --resume-existing
```

For disk-bounded runs, `--model-work-dir` isolates engines by exact
model/workload binding. Suites for the same model do not share engines because
their datasets may resolve to different static shapes, optimization profiles,
or cache lengths. A per-model HF cache can still be shared across those suites.
`--engine-retention` and `--hf-cache-retention` accept `retain`,
`delete_on_pass`, or `delete_always`; the latter applies only with
`--hf-cache-mode per_model`. A shared HF cache can only be retained.
`--storage-root` rejects mutable paths outside the managed filesystem, and
`--minimum-free-space-gib` is checked before every binding.

```bash
python tools/trtmc_validate.py \
  --model qwen25vl-3b \
  --storage-root /runs \
  --model-work-dir /runs/work/accuracy \
  --engine-retention delete_on_pass \
  --hf-cache-mode shared --hf-cache-retention retain
```

The same CLI is the CI case entry point. Generate a machine-readable matrix,
then run one exact model/workload binding in each CI node:

```bash
python tools/trtmc_validate.py --all --dry-run
python tools/trtmc_validate.py gpt2-125m mmlu_continuation_parity \
  --output validation-artifacts
```

For configured consistency workloads, the case result is always written to
`<output>/<model>/<workload>/comparison.json`; `report.json` and `report.html`
are written at the output root. `report.json` is the public result contract;
`report.html` is a static renderer that fetches that JSON and does not own
counts, verdicts, metrics, or links. The same directory also contains the
renderer under `assets/` and the run environment snapshot under
`artifacts/run/environment.json`. Exit status `0` means reference consistency
passed, `1` means the case ran but validation failed, and `2` means CLI or
setup validation failed before the case could run. Requesting a model that is
explicitly marked not compared also writes
`<output>/<model>/not-compared/comparison.json` and returns `2`.

`run.json` records each runtime-visible GPU's model, UUID, and PCI bus address.
The report labels `CUDA_VISIBLE_DEVICES` as a process-local selector because a
container may renumber a host GPU to logical device `0`. Validation refuses to
start when it cannot resolve that selector to stable GPU identity, preventing
an ambiguous report from being published.

Dataset-backed workloads use the task-specific `sample_limit` declared in
`workloads.yaml`. Fast encoder and classification workloads use larger
slices, while generation-heavy image, video, and audio workloads use smaller
slices. Set a workload to `-1` for its complete dataset. The runner never pads
or repeats samples: when the configured limit exceeds the available count, it
runs the available samples. `report.json` and the `Samples` column of
`report.html` record the actual prepared sample count.

Suites with configured gates resolve to `gate_policy: blocking`. A suite with
no gates must declare `gate_policy: observation_only`; shadow analysis marks an
empty blocking policy invalid instead of silently describing it as valid.
Completed Accuracy results preserve the
resolved gate configuration and publish a non-blocking analysis under
`comparison.gate_evaluation` in `report.json`. The analysis uses the valid
paired-sample count when available and expands rate gates into required passes,
allowed failures, observed passes, and observed failures. For example,
`min_prediction_agreement: 0.98` requires 20/20 at 20 valid samples and 49/50 at
50 valid samples. Continuous metrics retain their numeric threshold and do not
claim an integer failure budget.

This analysis is shadow evidence: it is shown separately in the HTML Metrics
details and does not change the existing comparison status, traffic light,
qualification snapshot, or CI disposition. Unsupported gate names, unavailable
metrics, non-numeric values, and missing sample counts are explicit `issues` in
the analysis rather than implicit passes.

Audit the resolved policy inventory without running any model:

```bash
python tools/trtmc_validate.py --gate-census > gate-census.json
```

The deterministic JSON groups models that resolve to the same gate variant and
shows the workload-owned rationale, configured sample count, effective integer
threshold, and unresolved configuration items. A suite that has no selected
model or sample limit remains visible in the census even though it cannot
appear in a model run.

When a scorer produces one pass/fail outcome per sample, configure the batch
decision once:

```yaml
sample_acceptance:
  min_pass_rate: 0.98
  min_allowed_failures: 1
```

The batch allows
`max(floor((1 - min_pass_rate) * sample_count), min_allowed_failures)` failed
samples. Each scorer still owns the per-sample pass/fail rule (for example, an
IoU threshold). Missing sample outcomes invalidate the comparison; aggregate
gates such as task-accuracy drop remain independent. The resolved counts and
verdict are written to `report.json`; HTML only renders them.

Override the configured limit for one run, or request the complete dataset
explicitly:

```bash
python tools/trtmc_validate.py gpt2-125m --limit 100
python tools/trtmc_validate.py gpt2-125m --limit -1
```

The command creates a reference environment only when one does not already
exist, then prints the environment it used. Reference inference runs through
`tools/trtmc_reference.py`, outside the validation engine process. Its result is keyed by
the input slice and inference settings and reused from the shared reference
cache when the key already exists. TRTMC variants may declare the same
`reference_cache_identity` in their owner-local `validation.yaml` only when they use the
same reference model, prepared inputs, and inference contract. The explicit
identity lets those variants share one cached reference result without
weakening cache isolation for other models.

TRTMC bundles live in one shared validation engine directory. A required
rebuild removes the existing bundle and writes the replacement at the same
path; a failed replacement removes any partial bundle. Per-run result
directories therefore do not retain another copy of the bundle.

At completion the command prints the exact reference and TRTMC reproduction
commands, the per-model `comparison.json`, and the aggregate `report.html`.
Comparison runs through `tools/trtmc_compare.py`; validation-engine commands
are not part of the result or its reproduction contract.
Every model/workload binding must resolve to an independent reference
runner selected by the prepared dataset kind. A catalog-wide test rejects new
bindings that have no independent native reference runner.

Tasks whose outputs require a model-specific comparator use
`model_plugin_json`. The workload and prepared dataset remain task-owned; a
row selects the matching model manifest testcase only at execution time. The
reference, TRTMC runner, and comparator are invoked directly without calling
the E2E orchestrator. Array-valued outputs are persisted as artifacts so a
cached reference can be compared in later runs.

Prepare the fixed task datasets from public benchmark sources already staged
on the validation machine:

```bash
python tools/prepare_model_plugin_validation_datasets.py \
  --output-root /mnt/data \
  --flores-source /mnt/data/FLORES200_en_fr/flores200_en_fr_task_eval.json \
  --full-duplex-source /mnt/data/FullDuplexBench-v1.0-public \
  --mmlu-source /mnt/data/MMLU_Pro/mmlu_pro_dataset.json \
  --mmmu-source /mnt/data/MMMU_Pro_vision/mmmu_pro_vision_dataset.json \
  --seedtts-source /mnt/data/SeedTTS_en_meta/seedtts_en_meta.json
```

This writes the `mmmu-pro-vision`, `mmmu-pro-vision-square-448`,
`mmlu-generation-modes`, `flores200-en-fr`, `full-duplex-bench`, and
`seedtts-en-omni-audio` directories directly under `/mnt/data`. It also writes
`/mnt/data/trtmc_model_plugin_validation_manifest.json`, which lists only
those six managed directories and records each file's byte size and SHA256.
There is intentionally no aggregate `TRTMCValidation` directory. Wan workloads
read the existing root-level public `VBench` asset directly. Dev/QA machines
and NAS mirrors should copy the six directories without changing their
relative layouts and verify the manifest after transfer.

PersonaPlex's `full_duplex_bench_behavior_parity` workload uses a
separate, deterministic behavioral slice of the public Full-Duplex-Bench v1.0
asset. Prepare it from the complete 727-sample benchmark:

```bash
python tools/prepare_full_duplex_bench_validation.py \
  --source-root /mnt/data/FullDuplexBench-v1.0-public \
  --icc-distribution /mnt/data/FullDuplexBench-v1.0-public/icc_backchannel/all_data_distribution.json \
  --output-root /mnt/data/FullDuplexBench-v1.0-public/trtmc-validate-v1 \
  --samples-per-category 30
```

The resulting 150 samples contain 30 fixed SHA-ranked inputs from each of the
five benchmark categories. This fixed, balanced slice is the formal validation
contract; changing its size requires reviewing both the sampling uncertainty
and the metric-delta gates rather than silently changing `--limit`.

The source root must include its managed `DATASET_MANIFEST.json`; preparation
rejects a different upstream revision, subset count, or license declaration.

HF and TRTMC process the exact same normalized audio independently. Each
backend is then scored with pinned Full-Duplex-Bench TOR, backchannel frequency,
and JSD definitions. Validation gates the absolute backend delta at 0.10 TOR,
0.01 backchannel events/second, and 0.02 JSD. These are behavioral consistency
gates, not paper-score or semantic-content accuracy gates. The generated
manifest preserves the per-category source licenses; CANDOR and ICC subsets
remain non-commercial and subject to their upstream terms.
Prepared 24 kHz mono float WAVs use deterministic headers, and the scorer
verifies every prepared-audio SHA before evaluation.
The scorer is TRTMC-owned code following the metric definitions from
`DanielLin94144/Full-Duplex-Bench` revision
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`; the repository does not vendor or
execute the upstream evaluator source. Dataset use remains governed by the
per-category licenses recorded in the prepared manifest.
Because those aggregate gates are sized for 30 samples per category, the
validation engine rejects a reduced `--limit` before launching HF or TRTMC
instead of reporting a statistically unsupported pass.

PersonaPlex uses `full_duplex_bench_behavior_parity` as its normal Accuracy
workload. `full_duplex_bench_speech_parity` remains available as an explicit
five-sample diagnostic benchmark. It contains the original `000000` and
`000002` synthetic-interruption failures from issue #767 and provides
per-sample token, waveform, and vanilla reproduction evidence, but it is not
selected by full-matrix model checks.

LocateAnything grounding accuracy uses the public `lscpku/RefCOCO_rec`
dataset pinned at revision
`566810e1ad62821ed3c6ab569ea33d80f5bdb874`. Stage that exact Hugging Face
snapshot, then convert its `testA` split:

```bash
python tools/prepare_refcoco_validation_dataset.py \
  --source-root /mnt/data/RefCOCO_rec/raw/lscpku/RefCOCO_rec \
  --output-dir /mnt/data/RefCOCO_rec/unified \
  --split testA
PYTHONPATH=python:. python tools/validation/engine.py eval \
  --suite refcoco_grounding \
  --model locateanything-3b \
  --dataset /mnt/data/RefCOCO_rec/unified/dataset.json \
  --limit 20
```

The source dataset card does not declare a license. The converter records this
as `source_license_status: not-declared-by-source-card`; operators must verify
their right to use the staged images and annotations. The generated manifest
records the source repository, immutable revision, normalized box format, and
official LocateAnything single-instance prompt.

Every agreement or disagreement therefore means that both backends consumed
the aligned prepared inputs and produced outputs that were evaluated by the
declared comparator. A model without that complete contract stays in the
catalog with `not_compared_reason`. `--all` records it as a white **Not
compared** row without launching E2E, creating a reference environment, or
building an engine. Such rows make the aggregate report status `incomplete`;
they are not agreements, disagreements, execution errors, or attempted model
failures.

`--all --dry-run` keeps these models visible with `workload: null`,
`status: not_compared`, and the reason. CI matrix generation can select only
entries that contain a workload.

The public artifact is named **TRTMC Accuracy & Fidelity Qualification** because
it covers task accuracy as well as token, embedding, and numerical agreement.
For large datasets it records one dataset-run command and at most three
representative commands per backend. The first disagreement is preferred when
one exists. The HTML presents Metrics, Logs, Vanilla reproduction, and Commands
as separate entries. Log entries are relative links to published files, never
execution-machine paths or captured output tails.

When per-sample differences exist, the model row also shows up to 20 affected
samples. Each sample contains the exact input, both raw prediction records,
the comparison evidence, and native single-sample commands when the backends
provide them. The reference command invokes a standalone upstream-framework
entrypoint and the TRTMC command invokes the model executable directly; neither
command re-enters validation or comparison orchestration. The
complete set is written to `disagreements.jsonl`, while `comparison.json` and
`report.json` retain only bounded metadata.

Model-owned reference and TRTMC plugins record the subprocess command they
actually executed. These per-sample command logs stay in the model work
directory; the HTML includes only commands for disagreement samples. This
keeps a 1,000- or 10,000-sample report compact without reconstructing a command
after the failure.

For failed image, video, or audio samples, the report copies only the relevant
input/output media into that sample's `repro` directory. Image and video-frame
previews, playable video files up to the artifact size limit, and WAV/audio
controls are rendered next to the two result records. Passing samples do not
duplicate media.

The report keeps execution evidence and the public outcome separate:

- `execution`: whether the programs completed or errored;
- `comparison`: whether TRTMC agrees or disagrees with the reference;
- `validation`: the final pass, fail, or skipped result.

An unimplemented consistency contract uses `execution: not_run`,
`comparison: not_run`, and `validation: not_compared`.

Platform exclusions are removed before `report.json` is materialized and never
appear in its rows, counts, search data, or denominator. A selected terminal
case receives green, yellow, or red only after execution completed, both compute
precisions were recorded, and a valid comparison exists. Otherwise it receives
white (`No valid comparison`) with structured priority, failed stage, cause
domain, reason code, and a direct diagnostic-log link. Pending and running cases
have no traffic light. `Comparable results` counts only green/yellow/red;
`Operational coverage` reports comparable/selected and the white count.

## Precision contract

Native Transformers text, embedding, VLM, speech, and model-owned plugin
references use the model manifest's FP16, BF16, or FP32 base precision. An explicit
`--hf-dtype` must match an unquantized TRTMC model's base precision; validation
rejects a conflicting override before inference.

Quantized candidates must declare their unquantized reference precision in the
model testcase's validation configuration:

The persisted manifest field remains named `task_eval` for compatibility with
existing reference-cache keys and result artifacts. It is metadata only; no
validation command imports or executes a task-eval module.

```json
"precision": "bf16",
"quantization": {"format": "fp8"},
"task_eval": {
  "reference_precision": "bf16"
}
```

This means TRTMC FP8 with a BF16 base is compared with an unquantized HF BF16
reference. It is a quantization-quality comparison, not an assertion that HF
executed FP8 kernels. The same contract applies to FP4, NVFP4, MXFP, or future
quantization formats when those candidates are added. A quantized manifest
without `task_eval.reference_precision` fails before reference inference.

An unquantized model whose official reference cannot execute at the TRTMC base
precision must declare the reviewed mismatch explicitly. For example, FNet
keeps its shipping candidate in FP16 while its non-power-of-two PyTorch cuFFT
reference runs in FP32:

```json
"precision": "fp16",
"task_eval": {
  "reference_precision": "fp32",
  "allow_reference_precision_mismatch": true
}
```

Without that explicit declaration, native unquantized precision mismatches
fail before reference inference.

The resolved TRTMC base precision, quantization format, reference precision,
and comparison kind are stored in `comparison.json` and shown in the HTML
report. Reference cache keys include the effective reference dtype, so only
variants with the same reference computation can reuse an entry.

## Add or extend a model

1. Reuse or add a dataset workload in
   `tests/validation/workloads.yaml`. Dataset variants that can change build
   shapes or profiles require distinct workload IDs. Put its shared
   `sample_limit` on that suite.
2. Add the model to the suite binding in
   `python/tensorrt_model_connect/models/<family>/validation.yaml`.
3. Add the model under that binding's `qualification` mapping when model and
   all-model runs should include it.

A model may list multiple workloads; selecting that model runs all of them as
independent bindings. Callers can narrow a one-off run by passing one workload
after the model name. Models without a `qualification` entry are not presented
as reference-consistency coverage.

Do not use `e2e` as a validation workload.

<!-- Collaborative review anchor: batch 2. -->
