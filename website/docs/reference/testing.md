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
and the CLI binary. The remaining runtime evidence depends on the bundle path:
a native bundle needs its owning model and backend DSOs; an optimized-runtime
bundle must contain the qualified implementation metadata, integrity-bound
artifact tree, and embedded implementation DSO.

`tests/test_e2e.py` is the repository-wide compatibility entry point. Model
work should normally select the owning `tests/e2e/models/<family>/` tree so
collection, defaults, waives, artifacts, and impact remain model-local.

## Native manifest and optimized-provider contracts

Each `tests/e2e/models/<family>/MODEL.toml` declares that family's JSON
manifests. Each buildable **native** JSON manifest requires:

- `name`, `hf_id`, `bundle`, and `family`
- an exact family-owned `runtime_strategy`
- a `task_strategy` or a runtime-strategy mapping from
  `tests/runtime_strategy_matrix.yaml`
- a non-empty `testcases` array

Each testcase carries the request and oracle contract, including fields such
as `name`, `user_contract`, `ci_tier`, prompt/media inputs, reference-family
metadata, and thresholds. Fields shared by every testcase stay at manifest
level; testcase values override inherited defaults.

An optimized implementation has a separate evidence chain:

- family-owned `IMPLEMENTATION.toml`, including the private
  `libtrtmc_impl_*.so` factory identity
- an exact qualified profile under `profiles/*.toml`, binding the model
  revision, target, options, and semantic hash
- a matching model-owned `QUALIFICATION.*.toml` producer descriptor and its
  retained parity/performance artifacts
- a built bundle containing `optimized_runtime.json`, `implementation.json`,
  the integrity-bound artifact tree, and the embedded implementation DSO

`PipelineFactory` checks `optimized_runtime.json` before native strategy
dispatch. Consequently, an optimized bundle's public `runtime_strategy` may be
empty: its implementation ID and profile ID are the selection evidence, and it
does not load a native model or backend DSO through the strategy registry.

Do not copy a generic example and invent either a runtime strategy or an
optimized profile. Use the matching Python, runtime, and E2E descriptors as the
source of truth.

## Runtime evidence by bundle path

| Evidence | Native bundle | Optimized-runtime bundle |
| --- | --- | --- |
| Build/selection identity | Family `MODEL.toml` and exact `runtime_strategy` | `IMPLEMENTATION.toml` plus an exact qualified `profiles/*.toml` entry |
| Qualification authority | Model-owned E2E manifest and retained comparison artifacts | Matching `QUALIFICATION.*.toml` producer proof and retained parity/performance artifacts |
| Bundle dispatch | `config.json` and `runtime_strategy` | `optimized_runtime.json` and `implementation.json` |
| Runtime libraries | Owning `libtrtmc_model_*.so` and selected `libtrtmc_backend_*.so` | Exact embedded `libtrtmc_impl_*.so`; no native strategy/model/backend dispatch |
| Timing evidence | Provider-populated `setup_ms`, `prefill_ms`, and `decode_ms`, when available | Provider-populated phase timing when available; otherwise synchronized public-call wall time |

The CLI prints the phase fields returned by `TextResult`, but providers are not
required to expose every phase. A zero phase value can mean unavailable; for
example, the qualified Qwen Edge-LLM adapter deliberately reports zero
prefill/decode timing because its pinned downstream API has no trustworthy
split. Do not turn those zeros into latency or throughput claims. Use the
qualification performance runner or another synchronized wall-clock
measurement and label that metric explicitly.

## Choosing evidence

| Change | Minimum useful evidence |
| --- | --- |
| Python family plugin | Focused builder tests and one representative E2E case |
| Native runtime model DSO | Focused C++ tests, strategy/descriptor checks, backend-load evidence, and matching E2E |
| Optimized implementation | Implementation/profile/qualification contract tests, embedded-DSO host tests, and matching qualified E2E/performance artifacts |
| Shared runtime/config | Focused unit tests plus affected-model selection |
| Public C/C++ API | API/ABI tests and CLI smoke |
| E2E runner/comparator | Focused harness tests and representative artifact |
| Quantization | Builder checks plus model/modality parity and health evidence |
| Documentation commands | Parser/help check, path check, and execution where dependencies permit |

Compilation is not parity. A single-model E2E is not broad regression proof.
Performance results are not qualification evidence unless the run also records
the exact commit, hardware, inputs, artifacts, and comparison baseline.
