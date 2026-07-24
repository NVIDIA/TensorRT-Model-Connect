# Family-Owned Optimized Runtime Adapter Implementation Plan

Status: In progress

:::caution Proposal, not current runtime support

This document is an implementation plan. Unchecked exit criteria are not
claims of support or qualification, and the example A100/Edge-LLM commands
require the private dependencies and target hardware named in the plan.
Current supported runtime behavior is defined by model-owned descriptors and
the normal native/delegated runtime tests, not by the desired end state below.

:::

Goal reference:
`website/docs/context/optimized-runtime-family-adapter-plan.md`

## Goal

Keep the TensorRT Model Connect (MC) CLI, Python, C++, and C APIs unchanged,
while allowing a qualified model deployment to use a third-party optimized
runtime such as TensorRT Edge-LLM.

The source ownership unit is deliberately:

```text
one adapter = one model family + one optimized runtime

one profile = one qualified model + revision + target + engine configuration
```

Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B therefore share one Qwen x Edge-LLM
adapter and one runtime DSO. They differ only in declarative profile data and
generated engine artifacts. Flux x Edge-LLM, or Qwen x a different optimized
runtime, owns a separate adapter.

This is the required scalability boundary: developers working on different
family-runtime pairs do not edit each other's source, while adding another
qualified model within an existing pair is normally a profile-only change.

## Reused Foundation

The generic delegation host from PR #477 remains the foundation. It already
provides the runtime-neutral mechanics needed to:

- discover a model-owned adapter;
- invoke adapter probe and build entry points below unchanged public APIs;
- package delegated artifacts in an MC bundle;
- recognize, validate, and materialize a delegated bundle;
- load a model-owned runtime DSO and create an MC pipeline.

This work must extend that path only where adapter-root profile discovery
requires it. It must not add a second optimized-runtime framework.

## Non-Goals

- No public `auto`, `edge-llm`, or `native` selector.
- No public API signature or result-format changes.
- No central model x revision x GPU x runtime lookup table.
- No shared model translation adapter across unrelated families or runtimes.
- No adapter, DSO, CMake target, or runner per model size or profile.
- No Edge-LLM dependency for MC native-only builds.
- No A100 hardware qualification in ordinary premerge CI until a managed A100
  runner pool exists; integration remains covered by deterministic contract
  tests and normal native-path validation.
- No generic host knowledge of Qwen IDs, Edge-LLM requests, or engine layout.
- No new process-global compatibility framework or public/private pipeline ABI
  digest as part of this feature. Existing loader and private factory contract
  checks remain in place.
- No checked-in engines, benchmark output, logs, screenshots, or `evidence/`
  directory. CI retains such output as workflow artifacts.

## Ownership and Layout

Everything specific to Qwen x Edge-LLM stays in the Qwen-owned Builder,
Runtime, and Test trees:

```text
# Builder
python/tensorrt_model_connect/families/qwen/edge_llm_adapter/
├── IMPLEMENTATION.toml
├── dependency.lock
├── adapter.py
└── profiles/
    ├── qwen3-0.6b-a100-sm80-fp16.toml
    ├── qwen3-1.7b-a100-sm80-fp16.toml
    └── qwen3-4b-instruct-2507-a100-sm80-fp16.toml

# Runtime
src/runtime/models/qwen/edge_llm_adapter/
├── CMakeLists.txt
├── adapter.cpp
└── exports.map

# Test
tests/e2e/models/qwen/edge_llm_adapter/
├── QUALIFICATION.a100.toml
├── CMakeLists.txt
├── performance_harness.py
├── performance_runner.h and two private runner sources
├── run_a100_ci.sh
├── test_performance_contract.py
├── test_adapter.py
├── test_runtime_contract.py
└── test_a100_e2e.py
```

There is exactly one Python build adapter, one C++ runtime adapter, one
exported DSO, and one parameterized test harness for the three profiles.

### Generic MC Host Owns

- model-family resolution and adapter-root discovery;
- isolated adapter `probe` and `build` invocation;
- rejection of ambiguous authoritative matches;
- bundle manifest and artifact-integrity validation;
- delegated artifact materialization and reuse;
- runtime DSO loading, factory creation, lifecycle, and error boundaries.

The standalone generic hardware-qualification dispatcher is also
runtime-neutral. It discovers
`QUALIFICATION.*.toml` descriptors in model-owned Test directories, maps an
exact Git diff to a hardware qualification matrix, and invokes the declared
model-owned entrypoints. It contains no Qwen, Edge-LLM, model-ID, profile, or
GPU table. Ordinary premerge CI does not call this workflow while there is no
managed target-hardware runner pool. A new family-runtime integration adds its
own descriptor and runner without adding another top-level workflow.
Qualifications sharing one downstream `runtime_id` elect exactly one representative.
Family-owned changes select that family only; shared host, bundle, build, or
dispatcher changes select the representative once per optimized runtime
instead of fanning out across every family.

### Qwen x Edge-LLM Adapter Owns

- loading and strictly validating its own profile files;
- matching model, resolved revision, deployment request, and target capability;
- translating MC build inputs to the pinned Edge-LLM builder;
- validating and returning `engine.dir`;
- translating MC text requests and Edge-LLM responses;
- constructing and retaining one long-lived Edge-LLM runtime per loaded MC
  pipeline.

The Edge-LLM repository and team remain independent of MC. Edge-LLM does not
import MC code or produce MC-specific artifacts.

## Adapter and Profile Contracts

The adapter root identifies the family-runtime pair:

```toml
schema_version = 2
implementation_id = "qwen.tensorrt-edge-llm"

downstream_runtime = "tensorrt-edge-llm"
downstream_version = "0.9.0"
downstream_commit = "<pinned-commit>"

[build]
entrypoint = "adapter.py"
timeout_seconds = 21600

[runtime]
library = "libtrtmc_impl_qwen_tensorrt_edge_llm.so"
abi = 1
```

The generic manifest contains only the launch, bundle, provenance, and runtime
ABI facts that MC must understand. The owning family is already established by
the discovery path, while profile location and schema remain private to the
adapter.

Each profile identifies one exact qualified deployment:

```toml
schema_version = 1
profile_id = "qwen3-1.7b-fp16--a100-pcie80-sm80--edgellm0.9-trt11.1"
qualification_state = "qualified" # or "candidate"
qualified_semantic_sha256 = "<sha256>" # required only when qualified
operation = "text-generation-v1"
precision = "fp16"
quantization = "none"
max_input_length = 1024
max_cache_length = 4096
max_batch_size = 4
minimum_memory_mib = 80000

[model]
id = "Qwen/Qwen3-1.7B"
revisions = ["<exact-revision>"]
architectures = ["Qwen3ForCausalLM"]

[target]
os = "linux"
architecture = "x86_64"
platform_kind = "discrete"
gpu_architecture = "sm80"
gpu_name = "NVIDIA A100 80GB PCIe"

[engine]
model = "qwen3"
# Exact dimensions and Edge-LLM engine metadata.

[builder]
max_input_len = 1024
max_kv_cache_capacity = 4096
max_batch_size = 4

[artifacts]
required_files = ["config.json", "llm.engine", "embedding.safetensors",
                  "tokenizer.json", "tokenizer_config.json",
                  "processed_chat_template.json"]
```

Only a `qualified` profile is eligible for automatic delegation. A
`candidate` profile records work that has not passed promotion gates and is not
selected by the normal MC path. Each qualified profile pins one semantic digest
covering its canonical profile data, adapter manifest and source, dependency
lock, and the exact runtime-source allowlist. Probe and build both recompute the
digest. A semantic source change therefore removes the stale profile from
delegation until the family-runtime pair is requalified and the pin is updated;
it also detects a change between probe and build.

`max_batch_size` is the built engine's capacity. It does not imply that MC has
a public text-batch API. The current CLI and pipeline text interfaces submit
one prompt per `generate()` call.

## Selection Semantics

The adapter probes the complete deployment identity rather than consulting a
global lookup table:

```text
model ID + resolved revision + deployment options + target capability
                              │
                              ▼
                  Qwen adapter profile match
                  ┌───────────┴───────────┐
                  │                       │
       exactly one qualified match      no match
                  │                       │
                  ▼                       ▼
          Edge-LLM delegated build   existing MC native path
```

- A wrong revision, unsupported target, or `candidate` profile is simply no
  optimized-runtime match. MC continues to its native builder when that
  deployment is natively supported; otherwise it returns the existing
  unsupported-deployment error.
- Multiple matching profiles, or multiple adapters claiming the same
  deployment, are configuration errors.
- Once one qualified profile is selected, its build is authoritative. A build
  failure is returned and must not silently fall back to native.
- A delegated bundle is permanently identified as delegated. If it is later
  loaded on an incompatible GPU, runtime loading fails before generation; it
  does not reinterpret that bundle as native.

### Existing Default-Tuple Ambiguity

The unchanged public build API supplies concrete deployment defaults even when
the caller writes only a model ID:

```text
precision=fp32, max_cache_length=256, max_batch_size=1
```

It therefore cannot distinguish an omitted tuple from a caller who explicitly
provided those same values. To preserve the required model-ID-only experience,
the Qwen Edge-LLM adapter treats that complete default tuple as default intent
and maps it to the qualified Edge-LLM engine tuple:

```text
precision=fp16, max_cache_length=4096, max_batch_size=4
```

The adapter also accepts that complete qualified tuple when stated explicitly.
It rejects partial or mixed tuples instead of guessing. This rule is private to
the Qwen x Edge-LLM adapter, is covered by contract tests, and does not change
public API signatures. If the product later needs to distinguish omitted from
explicit defaults, that requires a separate public API decision.

## Build-Time Control Transfer

```text
unchanged MC build API
        │
        ▼
resolve model family, revision, and requested/current target GPU
        │
        ▼
probe Qwen-owned adapters and match qualified local profiles
        │
        ├── no match ──► existing MC native builder
        │
        ▼
Qwen Edge-LLM adapter lazily resolves the pinned Edge-LLM dependency
        │
        ▼
translate MC request and invoke the official Edge-LLM build path
        │
        ▼
validate engine.dir and build a single Qwen Edge-LLM adapter DSO
        │
        ▼
package engine.dir + adapter DSO + Edge-LLM plugin in the MC bundle
```

Edge-LLM source and build work occur only after a qualified profile matches.
The exact Edge-LLM release/commit, TensorRT, CUDA, compiler, and target identify
the dependency build cache. A native-only MC request neither fetches nor builds
Edge-LLM.

The x86 release cohort is TensorRT 11.1.0.106 with CUDA 13.4; the existing
aarch64 release remains on TensorRT 11.2.0.113. The wheel metadata selects the
TensorRT package by architecture. Model Connect, the Qwen adapter, and Edge-LLM
must use the same process-wide TensorRT/CUDA ABI. The bundle therefore does not
carry a second `libnvinfer` or `libcudart`.

The x86 build and qualification image provides Python 3.12 with its `venv`
module, CMake 3.20 or newer, Ninja, GCC 13, the CUDA 13.4 toolkit, and the
TensorRT 11.1 SDK headers and libraries. These are environment prerequisites;
the model-owned adapter does not install or emulate them.

All three Qwen profiles package the same
`libtrtmc_impl_qwen_tensorrt_edge_llm.so` name and implementation identity;
their profile IDs, model revisions, dimensions, and `engine.dir` contents
differ.

## Runtime Control Transfer

```text
unchanged MC bundle-load API
        │
        ▼
validate delegated manifest and artifact hashes
        │
        ▼
materialize engine.dir into the runtime cache
        │
        ▼
load libtrtmc_impl_qwen_tensorrt_edge_llm.so
        │
        ▼
factory creates one long-lived Edge-LLM runtime for the selected profile
        │
        ▼
MC prompt/config ──► Qwen adapter ──► Edge-LLM execution
        │
        ▼
Edge-LLM result ──► Qwen adapter ──► unchanged MC TextResult
```

Materialization and runtime initialization happen during bundle loading, not
on every `generate()` call. MC does not duplicate Edge-LLM preprocessing,
scheduling, cache management, GPU memory management, or TensorRT execution.

API validation must reflect the APIs that exist today:

- CLI: build, inspect, load, and one text-generation request;
- Python: load and one text-generation request through the existing wrapper;
- C++: use the existing C++ CLI benchmark path to load one pipeline and issue
  consecutive `generate()` calls on that same long-lived instance. The A100
  test also compiles a temporary installed-SDK client that calls the unchanged
  `trtmc::load()` and `generate()` API; no model-specific runner is shipped;
- C ABI: create/load the pipeline, proving bundle validation, materialization,
  and DSO loading only. The current C ABI exposes neither text generation nor
  pipeline destruction; `trtmc_generate_batch` is image generation and is not
  an Edge-LLM text test. A validation probe therefore terminates after a
  successful create instead of inventing a new public API.

## Private Toolchain Compatibility

The model-owned DSO implements MC's private C++ factory contract and is loaded
into the same process as the MC core. Its compiler family/major version and
libstdc++ ABI must therefore be compatible with the MC SDK/core that loads it.

Keep this constraint private and simple:

1. Build the adapter DSO with the same selected compiler/toolchain identity
   used for the pinned Edge-LLM build and the private MC SDK headers.
2. Reuse the existing private factory/toolchain checks so an incompatible DSO
   is rejected before its factory creates a pipeline; do not add another ABI
   scheme in this feature.
3. Add an installed-wheel E2E on the supported x86 environment. Building and
   loading the DSO against only an in-tree build is insufficient proof.

This feature does not introduce a new public ABI, process isolation scheme, or
process-global compatibility registry.

## Implementation Work

The historical plan called for one focused replacement feature PR based on the
GitHub `main` branch. Remote names are checkout-local, so automation must
resolve and validate whichever remote points to the canonical GitHub
repository. The planned work was:

1. **Minimal generic discovery change**
   - Discover adapter roots only inside the already-resolved model family.
   - Preserve `implementation_id` as the family-runtime identity and
     `profile_id` as deployment identity.
   - Keep profile location, layout, and matching private to each adapter.
   - Carry the selected profile digest through probe/build binding.
   - Recursively package the one adapter root.
   - Do not add model/runtime names or profile tables to generic code.

2. **Consolidate Qwen x Edge-LLM**
   - Keep one `adapter.py`, one `dependency.lock`, one `adapter.cpp`, and one
     DSO.
   - Express 0.6B, 1.7B, and 4B only as profile TOML files.
   - Keep model dimensions and engine capacities in profile data.
   - Remove superseded per-profile adapters, DSOs, CMake targets, CI helpers,
     and duplicated runners.

3. **Harden only model-owned boundaries needed for real execution**
   - Lazy pinned dependency acquisition/build after selection.
   - Source architecture and artifact-layout validation.
   - Force CUDA device linking for Edge-LLM's whole-archived static core and
     reject unresolved DSO symbols at link time.
   - Toolchain compatibility check and installed-wheel loading proof.
   - Cold and warm dependency/runtime cache behavior.
   - No unrelated host infrastructure.

4. **Route tests by ownership**
   - Premerge: source-only integration, schema, matching, packaging, dispatch,
     and lifecycle contract tests; never Edge-LLM E2E on the GB300 pool.
   - Profile-only hardware qualification: that profile's schema/match test and
     target E2E after a managed target-hardware runner is available.
   - Qwen x Edge-LLM adapter/runtime change: all Qwen Edge-LLM profiles.
   - Shared Edge-LLM pin change: all adapters using that pin.
   - Generic discovery/bundle change: generic contracts plus one representative
     E2E per optimized runtime.

   Keep the generic `optimized-runtime-proof` workflow manual/reusable until a
   managed target-hardware runner pool exists; it is not a premerge dependency.
   Each model-owned descriptor declares its profile glob, exact target fields,
   digest-pinned container image, runner labels, trigger globs, and entrypoint.
   Once hardware qualification is enabled, a profile-only diff passes only that
   qualified profile basename to the model-owned runner. An adapter/runtime or
   shared-pin diff passes an empty profile filter, meaning every qualified
   profile for that descriptor's exact target. Profiles for another target are
   never collected by that job.

Adding a fourth Qwen x Edge-LLM profile must be demonstrable by adding one TOML
file and test data only; it must not require editing generic MC code,
`adapter.py`, `adapter.cpp`, CMake, exported symbols, or runners.

This profile-only rule applies within one compatible dependency cohort: the
same Edge-LLM, TensorRT, CUDA, compiler, and private MC factory ABI pin. If a
new profile requires an incompatible pin, this PR does not support keeping both
cohorts active. Updating the pin is a change to the one Qwen x Edge-LLM adapter
and forces every profile in that adapter to be requalified. Simultaneous
incompatible cohorts require a separate architecture decision; they must not
silently change this ownership granularity or make the adapter conditional on
multiple toolchains.

Malformed manifests and failed or timed-out probes have not claimed support,
so they are logged and isolated before selection; another sibling adapter or
the existing native path may still serve the request. After a probe returns a
positive support claim, build failure is terminal. After a delegated bundle is
created, validation, materialization, load, or execution failure is also
terminal. These are intentional ownership and fallback boundaries.

## Verification Plan

### Local and CPU Contract Tests

Run parameterized tests for all three profiles and verify:

- strict schema parsing and exact model/revision/target matching;
- `candidate` profiles are not selected;
- wrong revision and unsupported target produce no delegation and allow the
  existing native path;
- the complete MC default tuple and complete qualified tuple match, while
  partial/mixed tuples do not;
- ambiguous matches fail with a configuration error;
- probe/build profile-digest binding detects a changed profile;
- one adapter root and one common DSO are packaged for every profile;
- bundle path, symlink, artifact-hash, and cache-tamper checks reject invalid
  inputs;
- a synthetic fourth profile is discovered without production adapter, C++,
  CMake, or generic-host changes;
- model-family CI impact selection includes nested profile files and runs only
  the correct family scope.

### Real A100 End-to-End

On the x86 A100 80GB PCIe node, parameterize one E2E over Qwen3-0.6B,
Qwen3-1.7B, and Qwen3-4B:

1. Install the built MC wheel/package normally into the clean validation
   environment and verify it resolves TensorRT 11.1.0.106; do not use
   `--no-deps` or manually substitute another TensorRT/CUDA cohort.
2. Build from the unchanged MC CLI using the model ID and current target.
3. Inspect the bundle and verify the shared implementation ID/DSO plus the
   profile-specific ID, revision, and engine tree.
4. Run one CLI request and one Python request.
5. Run the existing `trtmc run --benchmark 2 --warmup 1` path, which loads one
   C++ pipeline and calls `generate()` repeatedly, proving long-lived reuse
   without adding a qualification runner.
6. Compile a temporary client against the installed C++ headers/library, call
   `trtmc::load()` and `generate()`, and keep two different Qwen profile
   pipelines alive in the same process while generating with both.
7. Use a small C ABI probe to create the pipeline and then exit, proving load
   and materialization without claiming C text generation or a destroy API.
8. Run Edge-LLM's official `llm_inference` executable directly against the
   same materialized `engine.dir`, prompt, and greedy generation settings;
   compare deterministic functional output with the MC delegated path.
9. Exercise cold and warm runtime-cache loads and verify reuse without artifact
   mutation.
10. Repeat the installed-wheel build/load path with the controlled compiler to
   prove private C++ ABI/toolchain compatibility.
11. Force one unsupported Edge-LLM deployment tuple through the native Qwen
    builder and runtime, proving that the process-wide x86 TensorRT 11.1 cohort
    preserves native execution as well as delegated execution.
The profile remains `candidate` if any required functional, accuracy,
packaging, lifecycle, or agreed performance qualification fails. Functional
and direct-runtime parity results must be reported separately from performance
promotion results; one does not substitute for the other.

Edge-LLM 0.9 does not expose a request-scoped, concurrency-safe prefill/decode
timing split. The adapter therefore reports those two MC fields as unavailable
instead of relabeling whole-request wall time. Qualification measures latency
and throughput externally around warm long-lived MC and direct Edge-LLM runs,
and must not treat the CLI's zero split fields as performance evidence. A
future downstream API that exposes request-scoped timing can be adapted inside
this model-owned DSO without changing the MC public API.

The promotion measurement uses the same engine, prompt, token limit, sampling
settings, CUDA synchronization, five warmups, thirty measured requests, and
three repetitions for both long-lived paths. MC must be within 1.05x of direct
Edge-LLM median latency, within 1.10x at p95, and retain at least 95% of direct
throughput. These ratios and the raw measurements are CI artifacts tied to the
exact tested clean Git revision or deterministic source-archive SHA-256. A
dirty Git checkout cannot promote a profile. The family-owned performance
runners are test-only sources: they are compiled by the A100 test, are not
installed, and are not part of any public API. A profile remains `candidate`
when any gate fails.

### CI Evidence

Ordinary premerge CI must run the source-only integration and model-owned
contract tests selected by the ownership rules. It must not dispatch an A100
job or attempt Edge-LLM E2E on the GB300 runner pool. The manual/reusable
hardware workflow remains dormant until a managed A100 runner pool exists.
When hardware qualification is enabled, upload command logs, comparison output,
and benchmark summaries as workflow artifacts. Do not add proof data to the
source tree or PR diff.

The model-owned runner must fail before building when the runner GPU does not
match the descriptor target, build and install a normal release wheel from the
exact clean revision, and clean root-owned container scratch through its
digest-pinned container. The generic workflow only selects, invokes, uploads,
and gates; all family/runtime artifact formats and test logic stay in the model
Test directory. Profile-only changes run only the matching profile shard.
Adapter, runtime, or shared changes run the full target qualification.

## Exit Criteria

- [ ] Public CLI, Python, C++, and C API signatures are unchanged.
- [ ] PR #477 remains the only generic delegation foundation.
- [ ] Adapter identity is model family + optimized runtime; profile identity is
      the exact qualified deployment.
- [ ] Qwen x Edge-LLM owns one Builder adapter, one Runtime DSO, and one
      parameterized Test harness.
- [ ] Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B are three declarative profiles using
      that same adapter and DSO.
- [ ] Only `qualification_state = "qualified"` profiles dispatch by default.
- [ ] A qualified profile dispatches only while its pinned semantic source
      digest matches; any adapter, dependency, profile, or runtime-source change
      requires requalification and a new pin.
- [ ] Wrong revision or unsupported build target produces no delegation and
      preserves native fallback behavior.
- [ ] A selected delegated build failure and an incompatible delegated bundle
      load fail without silent native fallback.
- [ ] Model-ID-only build behavior, including the complete-default-tuple
      interpretation, is explicit and tested.
- [ ] `max_batch_size` is tested as engine capacity without claiming a CLI or C
      text-batch API.
- [ ] Edge-LLM is acquired/built only after profile selection and is absent
      from native-only dependency paths.
- [ ] x86 wheel metadata, dependency lock, bundle metadata, adapter/plugin ELF
      dependencies, and observed runtime agree on TensorRT 11.1.0.106 and CUDA
      13.4; no bundle-local TensorRT/CUDA copy is packaged.
- [ ] The same installed x86 wheel builds and executes one native Qwen bundle,
      explicitly validating the process-wide TensorRT cohort outside Edge-LLM.
- [ ] Bundle load materializes `engine.dir` and initializes one long-lived
      Edge-LLM runtime before generation.
- [ ] C++ proves consecutive text requests; C proves only its supported
      create/load/materialization behavior.
- [ ] Direct parity uses Edge-LLM's official `llm_inference` path and the same
      engine and request settings.
- [ ] Ordinary premerge CI runs integration contracts only and does not invoke
      optimized-runtime hardware qualification.
- [ ] Installed-wheel A100 E2E proves the private C++ toolchain/ABI constraint.
- [ ] All three profiles pass real A100 build, bundle, load, and generation.
- [ ] Adding a fourth Qwen profile requires only profile and test-data changes.
- [ ] Qwen x another runtime and another family x Edge-LLM can be added in
      sibling family-owned directories without modifying this adapter.
- [ ] Adding another family-runtime qualification requires only its model-owned
      descriptor and runner; no new top-level workflow or premerge job is
      required.
- [ ] Profile-only CI selects only that qualified target profile, while
      adapter/runtime changes select all qualified profiles for the descriptor's
      exact target and never profiles for a different GPU.
- [ ] CI is green at the smallest correct ownership scope; validation output is
      stored only as CI artifacts and no `evidence/` directory is committed.
- [ ] Superseded per-profile adapters and duplicated infrastructure are removed.

The integration CI change is complete when the current source passes its
smallest correct unit/model scope without dispatching target-hardware E2E.
Hardware promotion remains a separate, deferred gate until a managed A100
runner pool exists and the current source satisfies the A100 criteria above.
