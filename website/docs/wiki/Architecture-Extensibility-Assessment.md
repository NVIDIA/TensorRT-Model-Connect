# Architecture Extensibility Assessment

Status: current mechanism assessment; supported models are defined by
descriptors and E2E evidence, not this prose page.

## Extension boundary

A checkpoint can reuse an existing family identity only when its configuration
and user-visible task contract match that family. Reusing the native
implementation further requires a compatible weight layout, TensorRT graph,
bundle contract, and runtime state. Otherwise choose either a new native family
across the Python, Runtime, and E2E trees or an exact delegated optimized
implementation for the existing family; the two ownership contracts are not
interchangeable.

There are two distinct extension routes:

- Native support owns three descriptors:
  `python/tensorrt_model_connect/families/<family>/MODEL.toml`,
  `src/runtime/models/<runtime-owner>/MODEL.toml`, and
  `tests/e2e/models/<family>/MODEL.toml`. It also owns a concrete
  `runtime_strategy` and `libtrtmc_model_<owner>.so`.
- A delegated optimized implementation for an existing family owns a
  family-local `IMPLEMENTATION.toml`, exact `profiles/*.toml`, an isolated
  adapter and `libtrtmc_impl_*.so`, plus
  `tests/e2e/models/<family>/<adapter>/QUALIFICATION.*.toml`. It does not need a
  synthetic native runtime strategy or native runtime descriptor merely to
  represent that implementation/profile.

Reusing a generic task shape does not imply reusing a native runtime strategy.
`task_strategy` selects the runner/comparator contract; `runtime_strategy`
selects a concrete native model DSO. An optimized bundle instead identifies its
exact implementation/profile with `optimized_runtime.json`.

## Cost by kind of change

| Change | Expected ownership |
| --- | --- |
| Another native checkpoint with an identical family contract | Native E2E manifest data and focused evidence |
| Exact optimized deployment tuple for an existing family | Family-local implementation/profile data, isolated adapter/runtime DSO, and producer qualification evidence |
| New weight/config variant within a family | Python family plugin and tests; runtime only if bundle/state changes |
| New graph semantics | Family-local builder/checkpoint logic and parity evidence |
| New runtime state or operation | Family-owned C++ plugin/pipeline and C++ plus E2E tests |
| New reusable task contract | E2E runner/comparator/threshold registration |
| New shared infrastructure | Shared code plus broad impact proof; must remain model-independent |

## Current mechanism

- Python family discovery is descriptor-first. It scans
  `python/tensorrt_model_connect/families/*/MODEL.toml`, uses aliases, prefixes,
  architecture patterns, and diffusion pipeline classes to import narrow
  candidate packages, and checks those candidates first. When descriptor
  candidates cannot resolve a full config, `find_plugin()` preserves a
  compatibility fallback: `pkgutil.iter_modules()` imports every non-private
  family module/package and runs its matching predicates.
- Native Runtime CMake scans `src/runtime/models/*/MODEL.toml`; those model DSOs
  register unique strategy keys with `PipelineRegistry`.
- Native E2E discovery scans `tests/e2e/models/*/MODEL.toml`.
- After family resolution, optimized build dispatch scans
  `IMPLEMENTATION.toml` only inside that selected family. Exactly one qualified
  profile may claim the model revision, active target, and effective options.
  One claim writes `optimized_runtime.json` and embeds the implementation DSO;
  no claim continues to the native builder.
- `tools/model_ci.py` validates ownership and creates isolated model
  projections.
- `tools/test_impact.py` selects affected models and task coverage.

No manual edit to a factory switch or central plugin target list is required.
Once an optimized adapter has claimed a request, its build or bundle-load error
is terminal rather than a fallback to native dispatch.

## Support claims

“Implemented” means source exists. “Tested” requires relevant current tests.
“Parity-qualified” requires comparison artifacts for the exact checkpoint and
configuration. “Performance-qualified” additionally requires exact hardware
and benchmark evidence. For an optimized route, qualification is bounded by the
exact model ID, immutable revision, target, public options, profile state, and
its `QUALIFICATION.*.toml` producer proof. Use
[Model Support](../getting-started/model-support.md) and the owning descriptors
for user-facing coverage; do not infer qualification from an architecture class
or an `IMPLEMENTATION.toml` alone.

## Verify an extension

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
python3 tools/ci/optimized_runtime_qualifications.py --all
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_model_plugin_encapsulation_static.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_optimized_runtime_qualifications.py \
  tests/builder/test_manifest_validation.py \
  tests/builder/test_optimized_runtime_orchestrator.py \
  tests/builder/test_optimized_runtime_capsules.py -q
```

Native model/GPU claims additionally require the selected E2E manifest on its
declared environment. Optimized claims require the entrypoint and
digest-pinned environment declared by the matching `QUALIFICATION.*.toml`;
host-only contract tests do not replace that producer run.
