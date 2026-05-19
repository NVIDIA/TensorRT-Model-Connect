---
title: Edge-LLM Submodule Integration
---

# TensorRT Edge-LLM Submodule Integration Plan

TensorRT Edge-LLM is a runtime-provider implementation for deployment
specialization. It can own the full request loop for models where Edge-LLM has a
better platform-specific runtime than Model-Connect's native TensorRT pipeline.

The user-facing contract should remain unchanged:

```bash
trtmc-build build <model> \
  --target <platform> \
  --set deployment.provider=tensorrt-edge-llm \
  -o model.trtfb

trtmc run model.trtfb --prompt "..."
```

The submodule is only a developer and CI reproducibility mechanism. It must not
change the bundle format contract, require users to manage an Edge-LLM engine
directory, or force every Model-Connect checkout to build Edge-LLM.

## Current State

The prototype integration treats Edge-LLM as an external source tree:

- CMake accepts `TRTMC_ENABLE_EDGE_LLM_PROVIDER=ON`.
- CMake accepts `TRTMC_EDGE_LLM_ROOT=/path/to/tensorrt-edge-llm`.
- Model-Connect finds Edge-LLM headers under `${TRTMC_EDGE_LLM_ROOT}/cpp`.
- Model-Connect links built Edge-LLM libraries such as `edgellmCore` and
  `edgellmTokenizer`.
- The runtime provider constructs Edge-LLM's `LLMInferenceRuntime` directly.
- The TensorRT Edge-LLM plugin DSO is loaded at runtime through
  `EDGELLM_PLUGIN_PATH`.
- The build provider can package an already-built Edge-LLM engine directory into
  `.trtfb` sections under `providers/edgellm/engine_dir/`.
- At inference time, Model-Connect materializes those bundle sections into the
  runtime cache and calls the Edge-LLM runtime library. It does not invoke the
  Edge-LLM inference binary.

The E2E proof used a copied Edge-LLM checkout inside the container at
`/tmp/tensorrt-edge-llm-codex`, built it there, then pointed
`TRTMC_EDGE_LLM_ROOT` and runtime library paths at that copied build. That is
valid for prototyping, but not a reproducible source integration strategy.

## Problem

External paths are flexible but weak for repeatable provider work:

- Agents cannot reliably infer which Edge-LLM commit to use.
- CI cannot reproduce the exact provider source without extra setup.
- Build fixes may accidentally live only in a temporary container copy.
- Runtime-provider work becomes harder to bisect across Model-Connect and
  Edge-LLM revisions.

A git submodule can solve source pinning and agent reproducibility, but it must
not turn Edge-LLM into an unconditional dependency for normal Model-Connect
builds.

## Decision

Add TensorRT Edge-LLM as an optional pinned submodule:

```text
third_party/tensorrt-edge-llm
```

Initial pin:

```text
20e9ff86492e121b2fdfb9165fef4b799f97f664
```

That commit is the known-good source revision used by the deployment
specialization prototype. Updating it is a deliberate provider-source change,
not an incidental cleanup.

Resolution order:

1. Use explicit `TRTMC_EDGE_LLM_ROOT` if provided.
2. Otherwise use initialized submodule `third_party/tensorrt-edge-llm` if it
   exists and contains the expected source layout.
3. Otherwise disable the provider or fail clearly when
   `TRTMC_ENABLE_EDGE_LLM_PROVIDER=ON`.

Default builds must keep working without initializing the submodule.

## Non-Goals

- Do not vendor generated ONNX directories, engine directories, TensorRT plans,
  runtime caches, or build outputs into git.
- Do not make Edge-LLM a mandatory dependency for native Model-Connect builds.
- Do not shell out to Edge-LLM's inference binary at runtime.
- Do not expose Edge-LLM implementation details in the basic user run command.
- Do not require users to clone or initialize the submodule to run a `.trtfb`
  unless their local `trtmc` binary was built without the provider DSO needed by
  that bundle.

## Target Architecture

Use the submodule as source input for an optional provider build.

```text
Model-Connect checkout
  third_party/tensorrt-edge-llm/      # pinned source, optional
  build/edge-llm/                     # generated Edge-LLM build artifacts
  build/libtrtmc_provider_edgellm.so  # preferred long-term provider DSO

model.trtfb
  deployment_manifest.json
  providers/edgellm/engine_dir/...
```

The runtime path should remain:

```text
trtmc run model.trtfb
  -> parse deployment_manifest.json
  -> select edge_llm variant
  -> materialize providers/edgellm/engine_dir/ into runtime cache
  -> load Edge-LLM provider adapter
  -> construct Edge-LLM LLMInferenceRuntime
  -> return TextResult through IPipeline
```

## Provider Boundary

The current prototype links Edge-LLM into `trtmc_core`. That works for proving
the runtime path, but it is not the best scalable shape. Edge-LLM static CUDA
libraries can pull CUDA device-link requirements into unrelated C++ test
binaries.

The scalable runtime boundary is a provider DSO:

```text
libtrtmc_provider_edgellm.so
```

The main runtime should discover and load provider DSOs through the deployment
provider registry. This keeps heavy provider dependencies out of `trtmc_core`
and avoids making every unit test link Edge-LLM.

The submodule plan has two implementation stages:

- Stage 1: optional submodule source root with the current direct-link provider.
- Stage 2: move Edge-LLM runtime provider into a dedicated provider DSO.

Stage 1 is acceptable only as a short bridge if default builds and unrelated
tests remain clean. Stage 2 is the scalable endpoint and should be the default
shape for this branch: `trtmc_core` materializes provider artifacts, loads
`libtrtmc_provider_edgellm.so`, and forwards generation through `IPipeline`.
Only the provider DSO links Edge-LLM headers/static libraries and performs CUDA
device linking.

## Build Resolution

CMake should derive the effective root with this logic:

```cmake
set(TRTMC_EDGE_LLM_ROOT "" CACHE PATH "TensorRT Edge-LLM repository root")

if(NOT TRTMC_EDGE_LLM_ROOT AND EXISTS
   "${PROJECT_SOURCE_DIR}/third_party/tensorrt-edge-llm/cpp/runtime/llmInferenceRuntime.h")
  set(TRTMC_EDGE_LLM_ROOT
      "${PROJECT_SOURCE_DIR}/third_party/tensorrt-edge-llm")
endif()
```

When `TRTMC_ENABLE_EDGE_LLM_PROVIDER=OFF`, CMake should not require the
submodule.

When `TRTMC_ENABLE_EDGE_LLM_PROVIDER=ON`, CMake should emit one of two clear
states:

- provider enabled with resolved include and library paths
- provider disabled or failed with exact missing headers/libraries and the
  command to initialize the submodule or pass `TRTMC_EDGE_LLM_ROOT`

## Build Commands

Initial developer flow:

```bash
git submodule update --init --recursive third_party/tensorrt-edge-llm

cmake -S third_party/tensorrt-edge-llm \
  -B build/edge-llm \
  <edge-llm-options>

cmake -S . -B build \
  -DTRTMC_ENABLE_EDGE_LLM_PROVIDER=ON \
  -DTRTMC_EDGE_LLM_BUILD_DIR=$PWD/build/edge-llm
```

If the submodule build output is standardized, Model-Connect can infer the
Edge-LLM build directory. If not, add explicit cache paths:

```bash
-DTRTMC_EDGE_LLM_BUILD_DIR=$PWD/build/edge-llm
```

Do not hide an expensive Edge-LLM build behind a normal Model-Connect configure
unless CI explicitly opts into that behavior.

Opt-in CI flow:

```bash
bash .github/scripts/run-gha-stage.sh edge-llm-provider
```

The GitHub workflow
`.github/workflows/edge-llm-provider.yml` is `workflow_dispatch` only. It checks
out the submodule recursively, builds Edge-LLM, builds
`libtrtmc_provider_edgellm.so`, packages a delegated bundle, runs
`trtmc inspect --deployment`, runs `trtmc run`, and archives the metadata,
inspect output, inference output, and benchmark log.

## Bundle Semantics

The submodule must not change what is stored in `.trtfb`.

The bundle should contain:

- `deployment_manifest.json`
- Edge-LLM runtime-provider variant metadata
- Edge-LLM engine directory files as bundle sections
- tokenizer/config/embedding files required by Edge-LLM runtime

The bundle should not contain:

- Edge-LLM git source
- Edge-LLM build directory
- generated ONNX export workspace unless the runtime actually needs it
- user-managed absolute paths

## Agent Workflow

Specialization agents should receive a bounded task:

```text
Task: add/update Edge-LLM provider for model X on platform Y
Writable ownership:
  - provider adapter files
  - provider build scripts
  - specialization tests
  - deployment manifest records
Read-only unless approved:
  - core bundle format
  - shared runtime factory
  - submodule commit pointer
```

Agents may update the submodule commit only as an explicit source-version change
with:

- old commit
- new commit
- upstream reason
- Edge-LLM build proof
- Model-Connect E2E proof
- rollback note

## Phased Plan

### Phase 1: Add Optional Submodule

- Add `.gitmodules` entry for `third_party/tensorrt-edge-llm`.
- Pin the submodule to a known-good commit.
- Document initialization and update commands.
- Ensure default configure/build works when the submodule is absent or
  uninitialized.

### Phase 2: CMake Root Resolution

- Keep `TRTMC_EDGE_LLM_ROOT` as the highest-priority override.
- Add fallback resolution to `third_party/tensorrt-edge-llm`.
- Add optional `TRTMC_EDGE_LLM_BUILD_DIR` if Edge-LLM build products are not
  always under `${TRTMC_EDGE_LLM_ROOT}/build`.
- Emit exact diagnostics for missing headers, core library, tokenizer library,
  TensorRT SDK, CUDA runtime, and Edge-LLM plugin DSO.

### Phase 3: Provider Build Isolation

- Keep Stage 1 direct linking working as a bridge.
- Move Edge-LLM provider implementation into a provider DSO.
- Teach the runtime provider registry to load `libtrtmc_provider_edgellm.so`.
- Keep `trtmc_core` free of Edge-LLM symbols when the provider is disabled.
- Keep unrelated C++ tests linkable without Edge-LLM CUDA static libraries.

### Phase 4: Build Provider Reproducibility

- Add a build-provider bootstrap path that can use the submodule's export/build
  tools.
- Keep explicit `deployment.edge_llm_engine_dir` for prebuilt engines.
- Add scripts or CMake presets for building Edge-LLM in CI and developer
  containers.
- Record the Edge-LLM commit and build metadata in provider build reports.

### Phase 5: E2E Validation

- Build a small supported model through Edge-LLM export and builder.
- Package the resulting engine directory into a `.trtfb`.
- Inspect the bundle deployment manifest.
- Run inference with `trtmc run` without passing a user-managed engine
  directory.
- Verify runtime cache materialization, non-empty output, a recorded correctness
  result, latency, throughput, and sampled peak GPU memory. The bootstrap
  provider-smoke comparator is `non_empty_text_output`; model-specific
  specializations should replace or extend it with their semantic comparator.

## Exit Criteria

This change is complete only when source pinning, optional build behavior, and
runtime delegation are all proven. Adding the submodule alone is not enough.

### Exit Criteria A: Source And Git Hygiene

- `.gitmodules` contains `third_party/tensorrt-edge-llm`.
- The submodule points at a documented, known-good Edge-LLM commit.
- A fresh clone can initialize it with:

  ```bash
  git submodule update --init --recursive third_party/tensorrt-edge-llm
  ```

- Default Model-Connect build and tests still work with the submodule absent or
  uninitialized.
- Generated Edge-LLM outputs are ignored and are not committed.

### Exit Criteria B: CMake Resolution

- `TRTMC_EDGE_LLM_ROOT` override still works.
- Initialized submodule root works when no override is provided.
- `TRTMC_ENABLE_EDGE_LLM_PROVIDER=OFF` does not inspect or require Edge-LLM.
- `TRTMC_ENABLE_EDGE_LLM_PROVIDER=ON` fails with actionable diagnostics when
  headers or libraries are missing.
- Configure logs print the effective Edge-LLM source root, build directory, core
  library, tokenizer library, and plugin DSO.

### Exit Criteria C: Provider Isolation

- Edge-LLM code is not linked into `trtmc_core` when the provider is disabled.
- Unrelated unit-test binaries do not require Edge-LLM CUDA device linking.
- If Stage 1 direct linking remains, the known limitation is documented and
  constrained to provider-enabled builds.
- The final scalable endpoint has Edge-LLM behind a provider DSO or an
  equivalently isolated provider boundary.

### Exit Criteria D: Bundle Contract

- A built Edge-LLM `.trtfb` contains `deployment_manifest.json`.
- `trtmc inspect --deployment` shows provider `tensorrt-edge-llm`, selected
  variant, target, compatibility, artifact section prefix, and fallback state.
- The bundle contains the provider engine directory as bundle sections.
- The bundle does not contain source checkout paths, generated ONNX workspaces,
  or absolute user-managed engine paths.

### Exit Criteria E: E2E Runtime

- A user can run:

  ```bash
  trtmc run edge_llm_delegated.trtfb \
    --prompt "The capital of France is" \
    --max-new-tokens 20
  ```

- The command constructs Edge-LLM through the provider adapter, not by invoking
  the Edge-LLM inference binary.
- The output is non-empty and writes a correctness result. For the bootstrap
  Edge-LLM provider smoke, `correctness.txt` records
  `comparator=non_empty_text_output`; model-specific provider tasks must add
  their own semantic comparator when one exists.
- Runtime materializes provider artifacts only into the internal/default runtime
  cache or the explicit `--runtime-cache` path.
- Benchmark output records latency, throughput, and sampled peak GPU memory.

### Exit Criteria F: CI And Reproducibility

- CI has an opt-in job that initializes the submodule, builds Edge-LLM, builds
  the provider, packages an Edge-LLM `.trtfb`, and runs inference.
- CI default jobs do not pay the Edge-LLM build cost.
- The opt-in job records:
  - Model-Connect commit
  - Edge-LLM submodule commit
  - TensorRT version
  - CUDA version
  - target GPU/platform
  - build command
  - inspect output
  - inference output
  - correctness result, such as `correctness.txt`
  - latency, throughput, sampled peak GPU memory, such as `benchmark.txt`

### Exit Criteria G: Rollback

- Reverting the submodule pointer or disabling `TRTMC_ENABLE_EDGE_LLM_PROVIDER`
  restores native Model-Connect builds.
- Existing `.trtfb` bundles without Edge-LLM deployment manifests continue to
  load through the default runtime path.
- Edge-LLM provider failures produce clear errors and do not corrupt native
  fallback execution.
