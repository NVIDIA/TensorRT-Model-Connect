---
title: Testing Reference
---

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
the CLI binary, and the owning runtime DSO.

`tests/test_e2e.py` is the repository-wide compatibility entry point. Model
work should normally select the owning `tests/e2e/models/<family>/` tree so
collection, defaults, waives, artifacts, and impact remain model-local.

## Manifest contract

Each `tests/e2e/models/<family>/MODEL.toml` declares that family's JSON
manifests. Each buildable JSON manifest requires:

- `name`, `hf_id`, `bundle`, and `family`
- an exact family-owned `runtime_strategy`
- a `task_strategy` or a runtime-strategy mapping from
  `tests/runtime_strategy_matrix.yaml`
- a non-empty `testcases` array

Each testcase carries the request and oracle contract, including fields such
as `name`, `user_contract`, `ci_tier`, prompt/media inputs, reference-family
metadata, and thresholds. Fields shared by every testcase stay at manifest
level; testcase values override inherited defaults.

Do not copy a generic example and invent a runtime strategy. Use the matching
Python, runtime, and E2E descriptors as the source of truth.

## Choosing evidence

| Change | Minimum useful evidence |
| --- | --- |
| Python family plugin | Focused builder tests and one representative E2E case |
| Runtime model DSO | Focused C++ tests, descriptor checks, and matching E2E |
| Shared runtime/config | Focused unit tests plus affected-model selection |
| Public C/C++ API | API/ABI tests and CLI smoke |
| E2E runner/comparator | Focused harness tests and representative artifact |
| Quantization | Builder checks plus model/modality parity and health evidence |
| Documentation commands | Parser/help check, path check, and execution where dependencies permit |

Compilation is not parity. A single-model E2E is not broad regression proof.
Performance results are not qualification evidence unless the run also records
the exact commit, hardware, inputs, artifacts, and comparison baseline.
