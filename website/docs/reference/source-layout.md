---
title: Source Layout
---

This page is a map of the current repository. Native model support is
deliberately split across three linked, model-owned descriptors:

| Path | Authority |
| --- | --- |
| `python/tensorrt_model_connect/families/<builder-family>/MODEL.toml` | Python family discovery, aliases, capabilities, and adapters |
| `src/runtime/models/<runtime-owner>/MODEL.toml` | Runtime DSO name, plugin entry points, strategy keys, config schemas, and C++ tests |
| `tests/e2e/models/<e2e-family>/MODEL.toml` | E2E manifests, model-local plugins, defaults, and test ownership |

Each directory name must agree with the `id` in its own descriptor. The three
physical names usually match, but their link is the exact
`runtime_strategy`, not filename equality: current builder/E2E owners
`magpie_tts` and `wan_t2v` map to runtime owners `magpie` and `wan`,
respectively. At this revision, all three trees contain 83 descriptors. The E2E
descriptors declare 220 JSON manifests; runtime descriptors declare 84 unique
strategy keys because one runtime owner exposes two strategies. Treat these
numbers as a checked snapshot, not a constant: the descriptor files are the
source of truth.

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
| `src/runtime/models/` | Family-owned runtime implementations and descriptors |
| `src/runtime/registry/` | DSO discovery, registry, and pipeline factory |
| `src/runtime/providers/` | Generic optimized-runtime descriptor, artifact, and private factory host |
| `src/tokenizer/` | Tokenizer implementations |
| `python/tensorrt_model_connect/` | Python build package |
| `python/tensorrt_model_connect/runtime_provider/` | Family-scoped optimized implementation discovery, isolated build, and generic bundle packaging |
| `tests/builder/` | Python builder tests |
| `tests/cpp/` | C++ runtime tests |
| `tests/e2e/` | E2E entry points and model-owned cases |
| `tests/e2e_harness/` | Manifest loading, orchestration, runners, and comparators |
| `tests/tools/` | Tests for repository tools |
| `tools/` | CI, comparison, profiling, and repository checks |
| `scripts/` | Scaffolding and operator utilities |
| `website/` | Docusaurus source |

## Runtime selection

For a native bundle, CMake scans `src/runtime/models/*/MODEL.toml`;
contributors do not maintain a central list of model plugins. At runtime,
`PipelineFactory` reads `runtime_strategy`, resolves the owning model DSO from
generated manifest data, loads that DSO, and asks `PipelineRegistry` for the
registered plugin.

For an optimized bundle, `PipelineFactory` first recognizes
`optimized_runtime.json`. `src/runtime/providers/optimized_runtime_host.cpp`
validates and materializes its embedded artifact tree, loads the exact
`libtrtmc_impl_*.so`, and asks its private factory to return an `IPipeline`.
The native strategy index, model DSO, and backend DSO are not part of that
path. Build-side implementation manifests and exact qualification profiles
live under the owning Python family; the current example is
`python/tensorrt_model_connect/families/qwen/edge_llm_adapter/`.

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

The broader runtime-strategy matrix command is a drift diagnostic:

```bash
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
```

At GitHub `main` commit
`e6b798cdb145c38caf1ede8eda7f5ce83f894138`, it exits nonzero because
`diffusion_sana_wm` is absent from the matrix and five speech/omni task entries
have no discoverable runner class. Report that known baseline separately from
new changes; do not present the command as a passing consistency check.

Use `tools/test_impact.py` for change selection. Do not infer ownership from an
old document count or from a removed shared runtime directory.

{/* Collaborative review anchor: batch 2. */}
