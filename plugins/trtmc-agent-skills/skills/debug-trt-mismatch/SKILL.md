---
name: debug-trt-mismatch
description: >-
  Diagnose a TensorRT-Model-Connect family whose native output disagrees with
  its declared reference or family-owned correctness contract.
---

# Debug a TensorRT Mismatch

Find the first family-owned boundary that diverges without changing the
workload or acceptance rule.

## Start from the owning case

Record the repository revision, checkpoint revision, family testcase, build
inputs, runtime root, hardware, seed, and request. Read these files before
changing code:

- `families/<family>/tests/manifests/<case>.json`
- `families/<family>/tests/thresholds/<case>.json`
- `families/<family>/tests/test_e2e.py`

Reproduce the exact direct E2E path:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/<family>/tests/test_e2e.py \
  --e2e-testcase <case> -vv
```

Provide the testcase's required `TRTMC_BINARY`, `TRTMC_RUNTIME_ROOT`, local
checkpoint, and GPU environment. Do not replace its reference, seed, or
threshold with a smaller smoke test.

## Localize inside the family

Separate build, native execution, reference execution, and comparison. Then
narrow the first failing boundary in the owning family: inputs and
preprocessing, weight mapping, graph output, runtime bindings or state,
postprocessing, or the family contract.

Reuse an existing family-local probe or test when one exists. Add a focused
family-local diagnostic only when the current evidence cannot identify the
boundary. Do not move model policy into core or import another family to share
debug code.

Never weaken a correctness gate to make the mismatch disappear. If the gate
is suspected to be wrong, demonstrate that independently and request review.

Report the exact reproducer, first divergent boundary, evidence that ruled out
earlier stages, smallest owner-local fix, and checks still not run.
