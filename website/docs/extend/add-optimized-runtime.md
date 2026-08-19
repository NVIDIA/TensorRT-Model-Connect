---
title: Add an Optimized Runtime Implementation
description: Add an exact-qualified family-owned delegated runtime without inventing a native strategy.
---

Use this route when an existing model family needs a delegated implementation
for one exact model, revision, target, and option tuple. This is not a generic
fallback backend and it does not add a synthetic native `runtime_strategy`.

The current Qwen TensorRT Edge-LLM adapter is the concrete example:

- builder capsule:
  `python/tensorrt_model_connect/models/qwen/edge_llm_adapter/`;
- private runtime implementation:
  `python/tensorrt_model_connect/models/qwen/runtime/edge_llm_adapter/`;
- Source-side adapter and runtime contracts:
  `python/tensorrt_model_connect/models/qwen/tests/edge_llm_adapter/`.

## 1. Keep ownership inside the family

Create a family-local adapter directory with:

```text
<family>/<adapter>/
├── IMPLEMENTATION.toml
├── adapter.py
├── dependency.lock
└── profiles/
    └── <exact-profile>.toml
```

`IMPLEMENTATION.toml` binds the implementation ID, downstream runtime version
and commit, isolated build entrypoint, private `libtrtmc_impl_*.so`, and private
factory ABI. The adapter must be reproducible from its pinned dependency
contract.

## 2. Declare exact profiles

Each profile must bind the facts that were qualified, including:

- immutable model revision and architecture;
- OS, CPU architecture, GPU architecture, and named target;
- operation, precision, quantization, input/cache/batch limits, and memory;
- engine metadata and required produced files;
- `qualification_state` and the semantic-source digest.

Do not use a profile to claim a family, GPU class, unpinned revision, or option
range that was not tested. More than one matching profile is an error.

## 3. Implement the isolated runtime DSO

The family-local C++ adapter builds the exact `libtrtmc_impl_*.so` named by the
implementation manifest. It exports the private factory ABI used by the generic
optimized-runtime host and returns a public `IPipeline`.

The DSO owns delegated-runtime interpretation. The generic host owns descriptor
validation, artifact hashing/materialization, identity checks, and lifecycle.
Host driver, CUDA, TensorRT, loader, and system-library dependencies remain
external to the bundle.

## 4. Separate Source contracts from target proof

Source owns the family-local `IMPLEMENTATION.toml`, `dependency.lock`, exact
profile TOMLs, adapter code, embedded implementation DSO, and fail-closed
contract tests. A profile's `qualification_state` and
`qualified_semantic_sha256` record the semantic source snapshot associated with
that profile; they are not a fresh target-hardware result by themselves.

The public repository does not publish a `QUALIFICATION.*.toml` descriptor,
hardware runner, or retained target artifacts for the current Qwen Edge-LLM
profiles. Target compatibility, model parity, and performance therefore need
separately retained external evidence tied to the exact model revision, profile
ID, semantic-source digest, implementation DSO, hardware, software cohort,
inputs, and result artifacts.

## 5. Validate Source contracts before external hardware proof

The adapter and runtime-contract tests compile a fake runtime. Run them after a
normal CMake configuration has provisioned
`build*/_deps/nlohmann_json-src/include`. If that dependency lives outside this
checkout, point
`_TRTMC_INTERNAL_QWEN_EDGE_LLM_NLOHMANN_JSON_INCLUDE_DIR` at its include
directory.

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate

PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_optimized_runtime_orchestrator.py \
  tests/builder/test_optimized_runtime_capsules.py \
  python/tensorrt_model_connect/models/qwen/tests/edge_llm_adapter/test_adapter.py \
  python/tensorrt_model_connect/models/qwen/tests/edge_llm_adapter/test_runtime_contract.py -q
```

These Source checks prove manifest, selection, packaging, identity, and
factory contracts. They do not prove model parity, target compatibility, or
performance. Run and retain those evidence layers in the controlled target
environment that owns them.

## Selection and failure semantics

The shared builder resolves the model family and calls its `model.build()`
once. A family that supports an optimized implementation owns the profile
decision inside that function. Exactly one qualified profile may claim the
request; no claim continues through that family's native recipe.

Once a profile claims a request, adapter build or optimized bundle-load failure
is terminal. It must not silently fall back to native behavior, because that
would invalidate the selected implementation and its qualification evidence.

See the
[optimized-runtime design record](../context/optimized-runtime-family-adapter-plan.md)
for rationale and historical implementation detail.

{/* Collaborative review anchor: batch 2. */}
