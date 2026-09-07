---
title: Testing Reference
---

# Testing reference

TRTMC tests follow the same ownership boundary as the implementation: a model
family owns its Python, C++, end-to-end, manifest, threshold, and reference
evidence under `families/<family>/tests/`. Shared tests cover only shared
contracts and mechanics.

## Fast repository checks

Run the structural validators before selecting expensive model tests:

```bash
PYTHONPATH=core/builder:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:. python3 tools/test_impact.py --validate
git diff --check
```

`model_ci validate` checks physical family ownership. `test_impact.py
--validate` checks that changed paths map to an unambiguous family or shared
test scope.

Run host-side Python tests with the builder package and repository root on the
import path:

```bash
PYTHONPATH=core/builder:. python3 -m pytest core/builder/tests tools/tests -q
```

After configuring a native build, run its compiled tests with CTest:

```bash
ctest --test-dir build --output-on-failure
```

Some tests require TensorRT, CUDA, GPUs, checkpoints, or family-specific
dependencies. Inspect CTest labels and the selected family manifest before
assuming that the entire suite is host-only.

## Family tests

A family may contain:

```text
families/<family>/tests/
├── test_e2e.py
├── test_*.py
├── cpp/
├── manifests/
├── thresholds/
└── data/
```

The exact contents are family-owned. There is no central `MODEL.toml`, runtime
strategy matrix, or model-to-plugin registry after the family isolation
cutover.

Use pytest selection from the family entry point:

```bash
export TRTMC_BINARY="$PWD/build/apps/cli/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build/install/lib"

PYTHONPATH=core/builder:. python3 -m pytest \
  families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16 \
  -q
```

The installed binary and runtime layout depend on the CMake configuration.
Use the manifest filename or the selectors shown by `pytest --help`; never
infer test support from a family directory alone.

## Documentation checks

Install the locked website dependencies, run its unit check, and build the
complete site:

```bash
npm --prefix website ci
npm --prefix website run test:model-support
npm --prefix website run build
```

The Docusaurus build verifies Markdown/MDX compilation, internal links, and
static assets. It does not prove that GPU commands or model outputs are
correct.

## Pull-request evidence

Public GitHub workflows run CPU and metadata checks on pull requests. Protected
GPU and legal validation is dispatched separately through the repository's
authorized Internal CI bridge. Keep raw protected logs and private environment
details out of public issues, pull requests, and documentation.

The Pages workflow builds and publishes the documentation site from `main`;
it is not a substitute for local or pull-request validation.

## Test design rules

- Put model semantics, thresholds, fixtures, and orchestration in the owning
  family.
- Put only model-agnostic contract tests in shared directories.
- Test the public build and task interfaces, not removed registry internals.
- Keep comparison thresholds meaningful. A failing criterion is evidence to
  investigate, not a reason to weaken the test.
- Record the model revision, manifest, target, TensorRT version, and produced
  artifacts for hardware-dependent validation.
