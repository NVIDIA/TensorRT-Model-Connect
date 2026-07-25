# Adding a Model Family

This page describes the current ownership contract. The maintained contributor
guide is [Add a Model Family](../extend/add-model-family.md).

## Required ownership roots

A family owns three directories with the same ID:

```text
python/tensorrt_model_connect/families/<family>/
src/runtime/models/<family>/
tests/e2e/models/<family>/
```

Each directory requires a `MODEL.toml`. Do not create a flat
`families/<family>.py` module and do not edit a central CMake plugin list.

## Python side

At minimum, provide:

- `MODEL.toml` with `id`, aliases/prefixes, and capabilities needed for
  discovery; `module` may remain as specialization/tooling metadata but does
  not select the runtime discovery import
- `plugin.py` with matching logic, checkpoint/config handling, the exact
  family-owned `runtime_strategy`, and the build entry point
- `__init__.py` exporting that package's `plugin`
- family-local config, checkpoint mapping, graph construction, and debug
  support required by that model

Python discovery indexes `families/*/MODEL.toml` and then follows one of three
flows:

1. A full config tries bounded `architecture_patterns` candidates before the
   compatibility fallback imports all non-private family modules/packages with
   `pkgutil` and runs their predicates.
2. A string or `model_type` tries a direct descriptor ID, then alias/prefix
   candidates, then the same all-package fallback.
3. A Diffusers pipeline class uses descriptor
   `diffusion_pipeline_classes` only and has no `pkgutil` fallback.

Descriptor routes import the family package and read the package-level `plugin`
exported by `__init__.py`; `module` is specialization/tooling metadata, not an
arbitrary runtime import selector. A flat module can therefore be observed by
the two compatibility flows, but it still does not satisfy the three-descriptor
support contract. Use an existing family with the same task and state shape as
a structural example, but do not import model-semantic helpers from an
unrelated family.

## Runtime side

`src/runtime/models/<family>/MODEL.toml` declares:

- `id`
- `runtime_library`
- `runtime_plugins` as `source.cpp|registration_symbol`
- `runtime_strategies`
- optional runtime config schemas and C++ tests

CMake scans these descriptors. The DSO registration function registers every
declared strategy with `PipelineRegistry`. Strategy names must be unique and
normally family-qualified, such as `gpt2_decoder_kv_cache`.

## E2E side

`tests/e2e/models/<family>/MODEL.toml` lists the JSON manifests and supplies
task defaults. Each buildable manifest needs the exact `runtime_strategy`, a
`task_strategy` (or a matrix mapping), and a non-empty `testcases` array.

Each testcase should state the user contract, CI tier, request, reference
oracle, thresholds, and a unique `trace_id` when it participates in formal
traceability.

## Validation

Replace placeholders before running the E2E command:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py

PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_plugin_encapsulation_static.py -q

PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/<family> \
  --e2e-model <manifest-name> \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

The last command requires the checkpoint, TensorRT/CUDA, a suitable GPU, the
compiled CLI, and the owning model DSO.

## Scaffolding status

`scripts/new_family.py --help` exists, but its current output does not create
the three required descriptors or all files imported by its generated plugin.
Until that script is upgraded and covered by an end-to-end scaffold test,
create the model-owned directories from a current working family instead of
treating the script as a complete onboarding path.
