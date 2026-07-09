# Issue #428 pure TensorRT reproducer

This reproducer removes `.trtfb`, `task_eval`, datasets, tokenizers, and the TRTMC C++ runtime from the failure path. It builds two serialized TensorRT plans directly from the Gemma graph and executes them with the TensorRT Python API and CUDA runtime bindings.

The failure is reproduced by this sequence:

1. Run the split prefill plan with the fixed token IDs `2,4521` (`Hello`).
2. Copy each prefill `present_k_N` / `present_v_N` output into a shared KV cache.
3. Bind that cache to the split decode plan.
4. Run one decode step.
5. Observe CUDA status 700 during decode synchronization, followed by a Myelin module-unload failure during TensorRT teardown.

Both plans run cleanly when tested separately. The minimal failing boundary is the prefill-produced KV state consumed by the decode engine.

## Validated environment

- TRTMC source: PR #413 at `a10ba64820082ad3d21dfc67bffe444dd95c9626`
- TensorRT: `11.2.0.113`
- CUDA image: `13.3`
- GPU: NVIDIA GB300
- Model: `google/gemma-2-2b-it`
- Precision: FP16
- KV cache length: `1741`

The model snapshot must already be accessible through the Hugging Face cache or normal Hugging Face authentication. Allow at least 16 GiB of free disk space for the two plans and logs.

## One-command reproduction

Run from this directory inside the TRT 11.2 environment:

```bash
./run_repro.sh
```

The first invocation builds the prefill and decode plans directly into `/tmp/trtmc-issue428-pure-trt`, then runs inference. Subsequent invocations reuse those plans. Override the location when needed:

```bash
./run_repro.sh --output-dir /path/with/enough/space
```

Expected final output on the affected TensorRT build:

```text
[pure-trt] gemma-2-2b-c1741-prefill: execute_sync_status=0
[pure-trt] prefill_to_decode_cache=OK
[pure-trt] gemma-2-2b-c1741-decode: execute_sync_status=700
[pure-trt] ISSUE_428_REPRODUCED decode_sync_status=700
[one-click] ISSUE_428_REPRODUCED
```

`run_repro.sh` returns zero when the known issue is reproduced. It writes `build.log` and `infer.log` under the output directory.

## Re-run inference without rebuilding

```bash
./run_repro.sh \
  --output-dir /path/with/existing/plans \
  --skip-build
```

## Verify a candidate fix

Use `--expect-fixed` after replacing TensorRT/Myelin with a candidate build:

```bash
./run_repro.sh \
  --output-dir /path/with/existing/plans \
  --skip-build \
  --expect-fixed
```

In this mode the wrapper returns zero only when decode and teardown complete cleanly without a CUDA/Myelin safety signature.

## Files

- `build_plans.py` builds standalone prefill and decode `.plan` files. It uses TRTMC only as the network-definition source; it never creates a `.trtfb` bundle.
- `infer_sequence.py` is the failing pure TensorRT/CUDA inference harness.
- `infer_engine.py` is a single-prefill-plan control used during minimization.
- `run_repro.sh` performs preflight, build, inference, log capture, and result classification.

## Current minimization result

| Path | Result |
| --- | --- |
| Decode plan with an empty cache | Clean |
| Prefill plan with `2,4521` | Clean |
| Prefill plan → shared KV cache → decode plan | CUDA 700; Myelin unload failure |
| Original standalone `trtmc run` | Exit 139 with the same teardown signature |

This isolates the failure from bundle parsing, tokenizer behavior, task-eval orchestration, dataset content, and the TRTMC C++ runtime.
