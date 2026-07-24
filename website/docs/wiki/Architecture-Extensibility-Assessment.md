# Architecture Extensibility Assessment

Status: current mechanism assessment; supported models are defined by
descriptors and E2E evidence, not this prose page.

## Extension boundary

A checkpoint can reuse an existing family only when its config, weight layout,
TensorRT graph, bundle contract, runtime state, and user-visible task contract
match that family. Otherwise it needs a new family-owned implementation in the
Python, Runtime, and E2E trees.

Reusing a generic task shape does not imply reusing a runtime strategy.
`task_strategy` selects the runner/comparator contract; `runtime_strategy`
selects a concrete model DSO.

## Cost by kind of change

| Change | Expected ownership |
| --- | --- |
| Another checkpoint with an identical family contract | Manifest/profile data and focused evidence |
| New weight/config variant within a family | Python family plugin and tests; runtime only if bundle/state changes |
| New graph semantics | Family-local builder/checkpoint logic and parity evidence |
| New runtime state or operation | Family-owned C++ plugin/pipeline and C++ plus E2E tests |
| New reusable task contract | E2E runner/comparator/threshold registration |
| New shared infrastructure | Shared code plus broad impact proof; must remain model-independent |

## Current mechanism

- Python family discovery scans
  `python/tensorrt_model_connect/families/*/MODEL.toml`.
- Runtime CMake scans `src/runtime/models/*/MODEL.toml`.
- E2E discovery scans `tests/e2e/models/*/MODEL.toml`.
- Runtime DSOs register unique strategy keys with `PipelineRegistry`.
- `tools/model_ci.py` validates ownership and creates isolated model
  projections.
- `tools/test_impact.py` selects affected models and task coverage.

No manual edit to a factory switch or central plugin target list is required.

## Support claims

“Implemented” means source exists. “Tested” requires relevant current tests.
“Parity-qualified” requires comparison artifacts for the exact checkpoint and
configuration. “Performance-qualified” additionally requires exact hardware
and benchmark evidence. Use [Model Support](../getting-started/model-support.md)
and the owning manifests for user-facing coverage; do not infer qualification
from an architecture class alone.

## Verify an extension

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_model_plugin_encapsulation_static.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/builder/test_manifest_validation.py -q
```

Model/GPU claims additionally require the selected E2E manifest on its
declared environment.
