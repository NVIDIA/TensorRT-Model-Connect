# Full-Matrix CI Throughput Experiments

- Status: active experiment; no merge approved
- Started: 2026-07-10
- Branch: `codex/ci-full-matrix-throughput-experiment`
- Base: `github/main` at `5c1fb98f60f3b41f017b54dda3c4451199cc2d34`

## Objective

Reduce the elapsed time of the full model-proof CI matrix without changing the
tested models, selected test cases, commands, arguments, acceptance thresholds,
or available runner/GPU resources.

No experimental commit is approved for merge until repeated full-matrix data
shows that it preserves the proof contract and improves throughput without
introducing starvation or material tail-latency regressions.

## Fixed Invariants

- The full matrix selects the same 77 models.
- Every model keeps its current C++, Python, and representative E2E selections.
- Every model performs exactly one guarded full engine build.
- Hermetic source projection, private model references, private Hugging Face
  cache copies, network isolation, DSO isolation, and proof artifacts remain
  required.
- Pass/fail criteria, time budgets, numerical thresholds, and fail-fast policy
  are not weakened to improve elapsed time.
- Runner count, GPU count, slots per GPU, CPU, memory, and storage are fixed.
- The branch remains unmerged until a human explicitly confirms the result.

## Baseline Evidence

Primary baseline:
[Actions run 29060715158](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29060715158),
tested merge revision `31708284cb786a84417d8400fe09620415b4054e`.

| Metric | Baseline |
| --- | ---: |
| Workflow elapsed | 1:35:34 |
| Model-matrix span | 1:33:24 |
| Selected/completed models | 77/77 |
| Successful model jobs | 77 |
| Sum of model-job wall time | 24:11:19 |
| Sum of core proof-step time | 22:13:16 |
| Sum of per-job image-ensure time | 1:43:53 |
| Mean model-job time | 18:51 |
| Median model-job time | 11:17 |

Historical comparison runs:

| Run | Result | Workflow elapsed | Notes |
| --- | --- | ---: | --- |
| [29034911003](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29034911003) | 77/77 passed | 1:28:13 | First observed successful 77-model run |
| [29040631532](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29040631532) | 76/77 passed | 1:35:05 | TimesFM failed; model phase was about 17 seconds slower than the primary baseline |
| [29057623769](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29057623769) | incomplete | 0:18:18 | Fail-fast: 18 passed, 1 failed, 15 canceled, 43 never started |
| [29060715158](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29060715158) | 77/77 passed | 1:35:34 | Primary baseline |

The primary baseline is about 8.3% slower than run 29034911003. The comparison
therefore needs repeated runs; a single successful run is not sufficient.

## Baseline Bottleneck Findings

Job duration is not equivalent to model execution duration. Raw logs from 69 of
77 baseline jobs showed approximately 8 hours 42 minutes of aggregate pre-lease
time, mostly overlapping GPU-lease waits. Representative splits are:

| Model | Proof step | GPU-lease wait | Approx. post-lease work |
| --- | ---: | ---: | ---: |
| qwen3_omni | 1:02:57 | 0:18:59 | 0:43:58 |
| wan_t2v | 1:08:16 | 0:47:48 | 0:20:28 |
| magpie_tts | 1:06:23 | 0:54:24 | 0:11:59 |
| phi_moe | 1:00:41 | 0:49:59 | 0:10:41 |
| nemotron_h | 0:54:30 | 0:46:31 | 0:07:59 |
| ltx_video | 0:50:29 | 0:38:38 | 0:11:51 |
| qwen3_5 | 0:49:09 | 0:40:06 | 0:09:03 |
| granite | 0:46:51 | 0:35:13 | 0:11:38 |

The largest real post-lease costs are qwen3_omni (about 43:58), sana_wm
(about 37:07), nemotron_speech_streaming (about 31:30), personaplex (about
21:29), wan_t2v (about 20:28), and gpt_oss (about 19:42). Engine compilation is
the main irreducible component for several of these models: qwen3_omni recorded
about 35:36 of TensorRT compilation, wan_t2v about 16:50 of engine build, and
gpt_oss about 14:35.

The matrix contains 67 shared-slot proofs and 10 exclusive-GPU proofs. The
current longest-first ordering uses stale total-job estimates and does not model
resource class or measured lease-hold time. PR
[438](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/438) addresses FIFO
fairness/starvation independently; throughput experiments must preserve that
fairness contract rather than replace it.

## Experiment Sequence And Gates

Each optimization is isolated in its own commit and full run so that its effect
can be attributed and reverted independently.

| ID | Change | Behavior change | Full run | Decision |
| --- | --- | --- | --- | --- |
| E0 | Structured proof-step, lease-wait, and lease-hold timing | None; observation only | Pending | Pending |
| E1 | Prepare and validate the CI image once per workflow, then pin its identity | Image lifecycle only | Pending | Pending |
| E2 | Resource-aware admission using measured lease-hold time and resource class, layered on FIFO fairness | Scheduling only | Pending | Pending |
| E3 | Move demonstrably CPU-only preparation outside the GPU lease | Lease boundary only | Pending | Pending |
| E4 | Content-addressed compiler cache, if scratch-build evidence supports it | Build reuse only; proof build remains required | Pending | Pending |

Promotion gate for an optimization:

1. The selection artifact still contains exactly the same 77 models and test
   cases.
2. All 77 proofs pass and retain exactly one guarded engine build per model.
3. Artifact/proof contracts and isolation checks remain unchanged.
4. Prefer at least three full runs for noisy scheduling changes.
5. Median full-matrix elapsed time improves by at least 10%; p95 model latency
   does not regress materially; no older request is starved.
6. Record commit SHA, run ID, selected-model count, outcome, workflow span,
   matrix span, per-model lease wait/hold, phase timings, and failures here.

## Measurement Contract

E0 records the following without changing execution order:

- `gpu-lease.json`: UTC request/acquire/release timestamps, monotonic lease-wait
  seconds, and monotonic lease-hold seconds.
- `model-proof-status.json`: UTC start/completion timestamps and monotonic
  duration seconds for each validation step.
- Existing Actions job/step timing remains the source for checkout and image
  preparation.

Monotonic durations are authoritative for elapsed time. UTC timestamps are for
cross-artifact correlation only.

## Local Validation Ledger

Validation at E0 before its first push:

| Check | Result |
| --- | --- |
| `bash -n .github/scripts/run-model-proof.sh` | Passed |
| `python3 -m ruff check tests/tools/test_model_proof_runner.py` | Passed |
| `python3 tools/legal_headers.py --check` | Passed: 0 findings |
| `pytest -q tests/tools/test_github_actions_ci.py` | Passed: 52 |
| Timing/lease focused model-proof tests | Passed: 3 |
| Full `tests/tools/test_model_proof_runner.py` | 69 passed, 2 failed before the new timing unit test was added |

Both full-suite failures were reproduced unchanged from the unmodified primary
worktree: the local `/tmp` filesystem rejects the test's mandatory reflink, and
the existing exclusive-reservation timing test allows the exclusive fake proof
to finish before the blocked shared proof requests its lease. They are retained
as local-environment/baseline failures, not silently excluded from the record.

## Run Ledger

| ID | Commit | Run | 77 selected | Result | Workflow | Matrix | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Baseline | `31708284cb786a84417d8400fe09620415b4054e` | [29060715158](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29060715158) | Yes | 77 passed | 1:35:34 | 1:33:24 | Primary comparison |
| E0 | Pending | Pending | Pending | Pending | Pending | Pending | Observation-only instrumentation |

For failed or canceled runs, retain the run in this ledger and record the exact
failed model/stage. Do not discard unfavorable measurements.
