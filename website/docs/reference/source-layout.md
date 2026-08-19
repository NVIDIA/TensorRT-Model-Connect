---
title: Source Layout
---

This page is a map of the current repository. Every supported model has one
physical owner and one descriptor:

| Path | Authority |
| --- | --- |
| `python/tensorrt_model_connect/models/<owner>/MODEL.toml` | Discovery metadata, runtime strategies and registrars, E2E manifests, and task defaults |
| `python/tensorrt_model_connect/models/<owner>/model.py` | Required Python build entry point |
| `python/tensorrt_model_connect/models/<owner>/runtime/CMakeLists.txt` | Native DSO sources, dependencies, warnings, optional kernels, and focused C++ test targets |
| `python/tensorrt_model_connect/models/<owner>/runtime/` | Native model DSO implementation |
| `python/tensorrt_model_connect/models/<owner>/tests/` | E2E manifests, assets, Python tests, and focused C++ tests under `tests/cpp/` |
| `python/tensorrt_model_connect/models/<owner>/tools/` | Optional owner-specific development tools |

The directory name and descriptor `id` must agree. Runtime identity is not a
second owner name: the DSO target and default library name derive from that
same ID. At this revision there are 82 descriptors, 215 JSON manifests, and 83
unique runtime strategy keys. Treat those numbers as a checked snapshot rather
than a constant; the owner directories are the source of truth.

## Top-level directories

| Path | Purpose |
| --- | --- |
| `include/trtmc/` | Public C++ headers, including the current C-linkage C++ subset in `pipeline.h`; this is not a C-compatible header or complete stable C ABI |
| `src/bundle/` | `.bundle` bundle parsing |
| `src/cabi/api/` | Implementation of the C-linkage C++ subset; it uses C++ types and currently has no pipeline-destroy entry point |
| `src/runtime/backend/` | Backend loading and implementations |
| `src/runtime/config/` | Runtime config schemas and layered resolution |
| `src/runtime/core/` | Model-independent device/runtime primitives |
| `src/runtime/domains/` | Small modality helpers shared across model DSOs |
| `src/runtime/registry/` | DSO discovery, registry, and pipeline factory |
| `src/runtime/providers/` | Generic optimized-runtime descriptor, artifact, and private factory host |
| `src/tokenizer/` | Tokenizer implementations |
| `python/tensorrt_model_connect/` | Python build package |
| `python/tensorrt_model_connect/runtime_provider/` | Family-scoped optimized implementation discovery, isolated build, and generic bundle packaging |
| `tests/builder/` | Python builder tests |
| `tests/cpp/` | Shared C++ runtime tests; model-specific C++ tests live with their owner |
| `tests/e2e/` | Shared E2E entry points and selection support |
| `tests/e2e_harness/` | Manifest loading, orchestration, runners, and comparators |
| `tests/tools/` | Tests for repository tools |
| `tools/` | CI, comparison, profiling, and repository checks |
| `scripts/` | Scaffolding and operator utilities |
| `website/` | Docusaurus source |

## Runtime selection

For a native bundle, CMake scans `python/tensorrt_model_connect/models/*/MODEL.toml`
and adds each owner's `runtime/CMakeLists.txt`; contributors do not maintain a
central list of model plugins or model sources. At runtime,
`PipelineFactory` reads `runtime_strategy`, resolves the owning model DSO from
generated manifest data, loads that DSO, and asks `PipelineRegistry` for the
registered plugin.

For an optimized bundle, `PipelineFactory` first recognizes
`optimized_runtime.json`. `src/runtime/providers/optimized_runtime_host.cpp`
validates and materializes its embedded artifact tree, loads the exact
`libtrtmc_impl_*.so`, and asks its private factory to return an `IPipeline`.
The native strategy index, model DSO, and backend DSO are not part of that
path. Build-side implementation manifests and exact qualification profiles
live under the owning model root; the current example is
`python/tensorrt_model_connect/models/qwen/edge_llm_adapter/`.

The generic task shape belongs in `task_strategy` (for example,
`text_generation_causal`). The `runtime_strategy` is the concrete runtime
contract and is normally family-qualified (for example,
`qwen_decoder_kv_cache`).

## Verify the layout

Run the repository-owned descriptor and focused contract checks:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_model_plugin_encapsulation_static.py \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py -q
```

The runtime-strategy control-plane command validates owner-local declarations:

```bash
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
```

It derives all 83 current native strategies from the 82 owner descriptors and
requires each strategy to map through owner manifests to exactly one shared
task contract, local runner/comparator coverage, and valid local diff checks.

Use `tools/test_impact.py` for change selection. Do not infer ownership from an
old document count or from a removed shared runtime directory.

{/* Collaborative review anchor: batch 2. */}
