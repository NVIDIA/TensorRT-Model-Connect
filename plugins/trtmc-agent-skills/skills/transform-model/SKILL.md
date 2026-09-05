---
name: transform-model
description: >-
  Add a Hugging Face or local checkpoint to TensorRT-Model-Connect as a
  self-contained family, or extend the family that already owns it.
---

# Transform a Model

Define the checkpoint, revision, task, target hardware, requested precision,
and required evidence. Read the upstream config and reference implementation,
then inspect the closest family only as a pattern.

Extend an owner only when checkpoint identity, weight mapping, graph dataflow,
native request lifecycle, and correctness contract genuinely match. Otherwise
create one new `families/<family>/` directory.

## Implement one vertical slice

- `support.py` recognizes exact model metadata and declares supported tasks
  plus one meaningful default. Zero or multiple owners are errors.
- `model.py` exposes one plain `build(request, writer)` function. It owns model
  config, weights, TensorRT graphs, precision, parallelism, and bundle section
  semantics. It must not inherit a builder or import another family.
- `runtime/CMakeLists.txt` builds `trtmc_model_<family>`. The family factory and
  pipeline implement the required abstract interface from `trtmc/task.h` and
  drive engines through the public backend contract.
- `requirements.txt` contains only optional packages needed by this family.
- `tests/` owns direct build/runtime E2E, manifests, thresholds, fixtures, and
  focused Python or C++ tests.

Keep model behavior inside the owner even when another family looks similar.
Do not modify core merely to reduce duplicated lines. If the requested user
behavior cannot be expressed by an existing Task interface, stop and describe
the required shared contract before expanding scope.

## Prove the smallest real path

First make one exact testcase work end to end:

```text
checkpoint -> support resolution -> family build -> bundle
           -> family DSO -> abstract Task API -> real output
```

Then run repository ownership checks and the selected family tests:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/<family>/tests --e2e-model <family> -q
git diff --check
```

Do not weaken the family oracle or add configuration for an unrequested future
case. Report the exact checkpoint, commands, results, hardware, and checks not
run before claiming support.
