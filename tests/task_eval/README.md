# Task Eval and Performance Evaluation

Task Eval keeps task/fidelity correctness separate from Performance Evaluation.
Existing suite behavior is unchanged unless `--performance-profile` is passed.

## ETTh1 observation run

```bash
python tools/task_eval.py eval \
  --suite etth1_time_series_parity \
  --model chronos-bolt-tiny-official \
  --dataset /path/to/ETTh1.csv \
  --engine-dir /path/to/bundles \
  --performance-profile etth1_process_e2e_observation_v1
```

The profile measures both HF and TRTFB. Its scope is `process_e2e`: each sample
includes child-process startup and model/runtime load. The Measurement Engine
uses an outer monotonic clock and does not aggregate the historical `wall_ms`
fields. GPU environment and memory sampling follow the first device selected by
`CUDA_VISIBLE_DEVICES` when it is set.

The model work directory contains:

```text
performance/
  hf_measurement.json
  trtfb_measurement.json
  measurements.jsonl
  performance_result.json
  baseline_candidate.json
```

`request_latency_ms.p50` is the median: half of included observations are less
than or equal to it. `p95` reports the tail. Warmup observations remain in the
raw artifact with `included: false` and do not enter either percentile.
The report also records mean, standard deviation, MAD, and the coefficient of
variation of per-process-repetition p50 values.

Every measured ETTh1 output is matched by sample ID and digest against the
already accepted correctness-stage output. Drift, missing output, or execution
failure increments `error_rate`, blocks Performance Evaluation, and makes the
baseline candidate ineligible for approval.

Observation mode never changes the existing task/fidelity result.

## Blocking workflow

Blocking requires a dedicated compatible runner and an explicitly approved
baseline from the exact blocking profile:

```bash
python tools/task_eval.py eval \
  --suite etth1_time_series_parity \
  --model chronos-bolt-tiny-official \
  --dataset /path/to/ETTh1.csv \
  --engine-dir /path/to/bundles \
  --performance-profile etth1_process_e2e_blocking_v1
```

The first run is `blocked` because no baseline exists, but it writes a baseline
candidate. Review its correctness prerequisite, workload identity, environment,
raw observations, dispersion, and metrics. Approval is an explicit human step:
the generated top-level and per-backend `eligible_for_approval` fields must
already be `true`; do not edit them. Set top-level `approved` and each approved
backend's `approved` to `true` in a reviewed baseline artifact. Then rerun with:

```text
--performance-baseline /path/to/approved-baseline.json
```

Profile digest, comparison key, workload, model, precision, backend, adapter,
and environment compatibility class must match. Any mismatch is `blocked`, not
silently compared. A correctness/fidelity failure also blocks Performance
Evaluation and cannot produce a passing baseline.
