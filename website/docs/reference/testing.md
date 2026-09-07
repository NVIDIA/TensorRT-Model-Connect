---
title: Testing Reference
description: Source, core, family, E2E, documentation, and performance validation boundaries.
---

Tests follow ownership. Shared tests prove shared contracts; each family owns
its model behavior and end-to-end evidence.

## Test map

| Scope | Location | Proves |
| --- | --- | --- |
| Python build core | `core/builder/tests/` | resolution, requests, bundle writing, graph transform |
| Native core | `core/runtime/tests/` | bundle bounds, loader, Task/Engine contracts, BYOK plumbing |
| Family units | `families/<family>/tests/` including optional `families/<family>/tests/cpp/` | family build/runtime policy |
| Family E2E | `families/<family>/tests/test_e2e.py` plus manifests | exact checkpoint-to-Task behavior |
| Tools and CI | `tools/tests/` | ownership, impact, packaging, and public gates |
| Benchmark app | `apps/benchmark/**/tests/` | measurement and report contracts |
| Documentation | `website/` inventory test and production build | generated routes, links, and rendering |

## Source-only validation

From the repository root:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 tools/test_impact.py --validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest -q
git diff --check
```

The public source-quality gate can be run against an explicit base revision:

```bash
python3 -m tools.community_ci source-quality --base github/main
```

That command requires the repository's public CI dependencies. It is not a GPU
or model-parity test.

## Native unit tests

After configuring and building the project in the supported environment:

```bash
ctest --test-dir build --output-on-failure
```

Inspect CTest selection and build options before assuming every compiled test
is CPU-only. Family CUDA or TensorRT tests require their owning target and
hardware.

## Run one family E2E

Each family test builds its own bundle, loads an isolated runtime containing
only core, the requested backend, and that family DSO, calls the real Task
interface, and applies the family-owned oracle.

```bash
TRTMC_BINARY="$PWD/build/trtmc" \
TRTMC_RUNTIME_ROOT="$PWD/build" \
PYTHONPATH=core/builder:apps/benchmark:. \
python3 -m pytest families/gpt2/tests/test_e2e.py \
  --e2e-testcase gpt2-125m \
  -q -x
```

Use the actual runtime output directory from your build. The selected family
may require its own `requirements.txt`, model assets, CUDA/TensorRT, and more
than one GPU. Do not substitute another checkpoint when a prerequisite is
missing.

Selectors are:

- `--e2e-model FAMILY_OR_MANIFEST`;
- `--e2e-testcase EXACT_CASE`; and
- `--e2e-models-file PATH` for newline-separated names.

Without an explicit selector or `TRTMC_E2E=1`, real family E2E tests skip by
design. A skipped case is not passed evidence.

## Premerge and nightly

Family manifests mark owner-selected fast cases with `premerge: true`. Nightly
may execute the complete declared set. There is no shared fallback comparator
or central replacement registry. See
[Premerge and Nightly Model Selection](e2e-l0-replacements.md).

## Documentation

```bash
npm --prefix website ci
npm --prefix website run test:model-support
SITE_URL=https://nvidia.github.io \
BASE_URL=/TensorRT-Model-Connect/ \
npm --prefix website run build
```

If diagram sources change, the website build also verifies that every
checked-in SVG matches its source.

## Evidence interpretation

- Compilation is not inference.
- A passed family E2E applies only to its exact manifest and testcase.
- A target-hardware result does not qualify another device or software cohort.
- A threshold must express the intended correctness contract and must not be
  loosened merely to make CI pass.
- Benchmark and profiling claims require separate reproducible evidence.
- Report pass, skip, deselection, and unrun boundaries separately.
