---
title: Deployment Specialization
---

# Deployment Specialization Plan

Deployment specialization is the product contract for running a model well on a
target platform. A user should ask for:

```bash
trtmc-build build <hf-model-or-local-dir> --target <platform> -o model.trtfb
./build/trtmc run model.trtfb --prompt "..."
```

The user should not have to choose whether the implementation is raw TensorRT,
Torch-TRT, FFI kernels, TensorRT Edge-LLM, or a future runtime. Those are
implementation variants selected by the builder and runtime resolver.

## Goals

- Keep the public user interface stable: build a `.trtfb`, run a `.trtfb`.
- Support platform-specific implementations without multiplying
  `runtime_strategy` values.
- Allow AI agents or specialization engineers to add targeted implementations
  with bounded ownership and clear validation requirements.
- Always preserve a default path unless a specialization explicitly documents
  that no portable fallback is bundled.
- Treat performance and memory as first-class outputs, not side notes.

## Non-Goals

- Do not make `runtime_strategy` encode platform or backend choices.
- Do not let specialization agents freely edit shared runtime/builder files as
  the normal path.
- Do not require users to manage provider-specific engine directories.
- Do not require all providers to be composable. Whole-runtime providers are
  selectable variants, not mixins.

## Core Model

Deployment specialization is represented as implementation variants inside the
bundle.

```text
model + target + objective
  -> build provider
  -> implementation variants
  -> .trtfb bundle
  -> runtime resolver
  -> selected IPipeline
```

Variants have different scopes:

| Scope | Example | Composition |
| --- | --- | --- |
| `kernel` | FFI attention kernel replacing native attention | Composable when tensor contracts match |
| `component` | Specialized encoder, decoder, VAE, tokenizer path | Partly composable |
| `runtime` | TensorRT Edge-LLM owns the full request loop | Selectable, not composable |

The common variant manifest should describe all three.

```json
{
  "schema_version": 1,
  "target": {
    "platform": "gb300",
    "objective": "best_perf_memory"
  },
  "default_variant": "portable_default",
  "selected_variant": "gb300_ffi_attention",
  "variants": [
    {
      "id": "portable_default",
      "scope": "runtime",
      "provider": "native_trt",
      "runtime_strategy": "decoder_kv_cache",
      "fallback": true,
      "artifacts": [
        {"name": "engine_plan", "kind": "bundle_section"}
      ]
    },
    {
      "id": "gb300_ffi_attention",
      "scope": "kernel",
      "provider": "tvm_ffi",
      "runtime_strategy": "decoder_kv_cache",
      "compatibility": {
        "platform": ["gb300"],
        "gpu_arch": ["sm100"]
      },
      "artifacts": [
        {"name": "kernel_manifest.json", "kind": "bundle_section"},
        {"name": "kernel_flash_attention.so", "kind": "shared_library"}
      ]
    },
    {
      "id": "jetson_thor_edgellm",
      "scope": "runtime",
      "provider": "tensorrt-edge-llm",
      "runtime_strategy": "text_generation",
      "compatibility": {
        "platform": ["jetson-thor"],
        "provider_abi": "edgellm-0.6"
      },
      "artifacts": [
        {"name": "providers/edgellm/engine_dir.tar.zst",
         "kind": "directory_archive"}
      ]
    }
  ]
}
```

## Build Providers

The current builder already resolves model metadata, finds a plugin, executes
plugin-owned code, and writes a bundle. Deployment specialization should lift
that into a provider-level interface.

```python
class BuildProvider:
    name: str

    def matches(self, request: BuildRequest) -> bool:
        ...

    def build(self, request: BuildRequest) -> BuildResult:
        ...
```

`BuildRequest` should include:

- model path or HuggingFace id
- resolved model config
- deployment target
- objective, initially `best_perf_memory`
- workload profile, such as sequence length, batch size, image size, or audio
  shape
- fallback policy

`BuildResult` should include:

- sections to write into the `.trtfb`
- implementation variants
- provider-specific reports
- validation artifacts
- performance and memory measurements

Existing family plugins become the `native_trt` build provider. FFI attention is
either a post-build specialization provider or a native provider extension.
TensorRT Edge-LLM becomes a runtime-provider build provider.

## Runtime Providers

Runtime providers adapt a provider-specific runtime into the stable
`trtmc::IPipeline` API.

```cpp
class IRuntimeProvider {
  public:
    virtual ~IRuntimeProvider() = default;
    virtual bool can_load(const ProviderManifest& manifest,
                          const DeploymentTarget& target) const = 0;
    virtual std::unique_ptr<IPipeline> load(const ProviderContext& ctx) = 0;
};
```

`PipelineFactory` should resolve variants before constructing a pipeline:

```text
ReadBundleFile
  -> parse config.json
  -> parse deployment_manifest.json
  -> resolve target and compatibility
  -> if selected variant scope == runtime:
       load runtime provider and return provider IPipeline
     else:
       prepare selected kernel/component artifacts
       create native runtime plugin as today
  -> if selected variant fails compatibility:
       use default_variant
```

Runtime providers own their internal request loop, KV cache, tokenizer behavior,
engine layout, and provider-specific assets. They must map from the public
Model-Connect API into provider-specific requests and map provider responses
back to Model-Connect result types.

## Artifact Store

All specializations should use one artifact layer.

```cpp
class ArtifactStore {
  public:
    ByteSpan read(std::string_view name);
    std::filesystem::path materialize(std::string_view name);
    std::filesystem::path materialize_directory(std::string_view archive_name);
};
```

The artifact store lets providers choose memory-backed loading when possible
and materialized-file loading when required.

TensorRT Edge-LLM currently expects an engine directory containing files such as
`.engine`, `config.json`, `embedding.safetensors`, tokenizer assets, optional
vocab maps, and multimodal subdirectories. The first implementation should
package that directory inside the `.trtfb` and materialize it into a
content-addressed runtime cache. A future provider API can remove that
materialization by accepting memory-backed artifacts.

## TensorRT Edge-LLM Delegation

The Edge-LLM build provider should initially use Edge-LLM's existing contracts:

```text
HF checkpoint
  -> Edge-LLM Python export
  -> ONNX directory
  -> Edge-LLM C++ LLMBuilder
  -> engine directory
  -> .trtfb provider variant
```

The first implementation can call scripts or command-line tools as a bootstrap.
The long-term implementation should call Python export APIs and C++ builder
libraries directly.

The runtime provider should link to Edge-LLM runtime libraries and construct the
provider runtime directly. It should not shell out to the Edge-LLM inference
binary.

The provider pipeline should map:

| Model-Connect API | Edge-LLM Runtime |
| --- | --- |
| `GenerateConfig.max_new_tokens` | `LLMGenerationRequest.maxGenerateLength` |
| prompt string | chat-style `messages` request |
| temperature/top-k/top-p | Edge-LLM sampling fields |
| `TextResult.text` | first response text |
| `TextResult.token_ids` | provider output ids |

Fallback should remain a separate native variant when feasible.

## FFI Attention Delegation

FFI attention remains a kernel-scope specialization. The build provider should:

- build the normal native TensorRT graph
- replace selected attention blocks with an FFI plugin when compatibility
  predicates match
- package kernel shared libraries and `kernel_manifest.json`
- declare the variant as `scope=kernel`
- keep the native TensorRT attention path as fallback

The runtime resolver should load the selected FFI kernels before constructing
the native pipeline. FFI kernel loading should be centralized instead of
requiring every runtime plugin to call it.

## Agent Workflow

Specialization work should be dispatched as bounded tasks.

```text
specializations/
  <model-or-family>/
    <platform>/
      manifest.json
      build.py
      runtime/
      kernels/
      tests/
      benchmark.json
      validation.json
```

Agents may own files under a specialization directory and provider-specific
adapter files. Shared framework changes should be limited to stable extension
points:

- provider registry
- deployment manifest parser
- artifact store
- runtime resolver
- FFI loader
- validation harness

Every specialization PR must include correctness, performance, memory, and
fallback evidence.

## Phased Plan

### Phase 1: Shared Variant Infrastructure

- Add deployment manifest schema.
- Add artifact store with bundle-section, file, shared-library, and
  directory-archive artifact kinds.
- Add build-provider result type.
- Add runtime-provider registry.
- Add `trtmc inspect --deployment` and `trtmc-build inspect --deployment`.
- Keep existing native builds working without a deployment manifest.

### Phase 2: FFI Attention Delegation

- Centralize FFI kernel loading in runtime factory or resolver.
- Add a kernel-scope FFI attention variant.
- Package FFI kernel artifacts into `.trtfb`.
- Validate fallback to native TensorRT attention.
- Record performance and memory deltas versus native attention.

### Phase 3: Edge-LLM Runtime Provider

- Add Edge-LLM build provider using script/tool invocation first.
- Package Edge-LLM engine directory as a directory archive inside `.trtfb`.
- Add Edge-LLM runtime provider that links to Edge-LLM libraries.
- Materialize provider engine directory into internal runtime cache.
- Adapt Model-Connect text generation calls to Edge-LLM request/response
  structs.
- Keep user-facing `trtmc run` unchanged.

### Phase 4: Direct API Integration

- Replace Edge-LLM script invocations with direct Python export API calls.
- Replace C++ build subprocesses with direct `LLMBuilder` library calls.
- Add optional memory-backed provider loading where provider APIs allow it.
- Move performance records into the common perf database keyed by target and
  variant id.

## Exit Criteria

This plan is complete only when both delegation paths work through the normal
Model-Connect interface.

Partial infrastructure does not satisfy the exit criteria. Synthetic bundles,
manifest-only inspection, provider library builds, or a runtime that only
materializes artifacts and then fails are useful milestones, but they are not
completion evidence. Completion requires real model build and inference runs
using the same commands a user would run.

### Exit Criteria A: Edge-LLM Runtime Delegation

A user can build and run a supported model through Edge-LLM delegation without
managing an Edge-LLM engine directory.

Required command shape:

```bash
trtmc-build build <edge-llm-supported-model> \
  --target <edge-platform> \
  --set deployment.provider=tensorrt-edge-llm \
  -o /tmp/edge_llm_delegated.trtfb

./build/trtmc inspect /tmp/edge_llm_delegated.trtfb --deployment

./build/trtmc run /tmp/edge_llm_delegated.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

Acceptance requirements:

- The bundle contains an Edge-LLM runtime-provider variant.
- The bundle contains the provider engine directory or a provider artifact
  archive sufficient to reconstruct it.
- `trtmc inspect --deployment` shows provider, target, selected variant,
  compatibility, and fallback state.
- `trtmc run` constructs the Edge-LLM runtime through a library adapter, not by
  invoking the Edge-LLM inference binary.
- The generated output is non-empty and passes the model's E2E correctness
  comparator.
- The provider artifact is materialized only into an internal cache or explicit
  runtime cache path, not a user-managed engine directory.
- If a native fallback is bundled, forcing fallback produces valid output from
  the same `.trtfb`.
- The recorded proof includes the exact model id, target platform, build log,
  inspect output, run output, runtime cache location, correctness result, and
  throughput/latency/peak-memory measurements.

### Exit Criteria B: FFI Attention Delegation

A user can build and run a supported decoder model with FFI attention selected
as a kernel-scope specialization.

Required command shape:

```bash
trtmc-build build <ffi-attention-supported-model> \
  --target <platform-with-ffi-kernel> \
  --set deployment.enable_ffi_attention=true \
  -o /tmp/ffi_attention_delegated.trtfb

./build/trtmc inspect /tmp/ffi_attention_delegated.trtfb --deployment

./build/trtmc run /tmp/ffi_attention_delegated.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20
```

Acceptance requirements:

- The bundle contains a kernel-scope implementation variant for FFI attention.
- The bundle includes `kernel_manifest.json` and all required kernel shared
  library sections.
- Runtime loads FFI kernels through the centralized resolver path.
- The native TensorRT attention variant remains available as fallback.
- FFI and native fallback outputs pass the same correctness comparator.
- Benchmark output records throughput, latency, and peak memory for both FFI
  and fallback variants.
- `trtmc inspect --deployment` identifies the selected FFI variant and fallback
  variant.
- The recorded proof includes the exact model id, target platform, build log,
  inspect output, FFI run output, fallback run output, correctness result, and
  throughput/latency/peak-memory measurements for both paths.

### Exit Criteria C: Common Specialization System

The two delegation demos must share infrastructure rather than separate
one-off paths.

Acceptance requirements:

- Edge-LLM delegation and FFI attention delegation both use the same deployment
  manifest schema.
- Both use the same artifact store.
- Both appear in the same inspect output.
- Both write performance and memory records with `target_id`, `variant_id`,
  `provider`, and `scope`.
- Adding a third provider or kernel specialization does not require changing
  public CLI run commands.
- Existing `.trtfb` bundles without deployment manifests still load through the
  default runtime path.

## Open Design Decisions

- Whether `--target` should be a first-class build flag or only a schema-backed
  alias over `--config` and `--set`.
- Whether provider artifacts should be compressed with `tar.zst`, uncompressed
  tar, or section-per-file layout.
- Whether provider DSOs should be linked into the main runtime binary or loaded
  dynamically.
- Whether fallback should be mandatory for release builds or optional for
  platform-only artifacts.
- How to expose provider-specific diagnostics without leaking backend details
  into the basic user workflow.
