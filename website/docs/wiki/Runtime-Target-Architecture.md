# Runtime Target Architecture

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-RT-001 |
| **Status** | IMPLEMENTED |
| **Applies to** | C++ runtime (`src/`) |
| **Author** | Safety Architecture Team (yifeif@nvidia.com) |
| **Reviewer** | Independent Review Required (TBD — assign before merge) |
| **Review Status** | Pending independent review |
| **Last updated** | 2026-03-30 |
| **ISO 26262 relevance** | ASIL-QM (non-safety, design improvement) |

---

> **STATUS: IMPLEMENTED**
>
> The plugin-registry architecture described in this document has been
> fully implemented. The `PipelineRegistry` singleton, `IPipelinePlugin`
> interface, manifest registration, and `BaseConfig` parsing
> are all in the codebase. `PipelineFactory::from_bundle()` is now a thin
> ~124 LOC wrapper that delegates to registry-resolved plugins.
> See [Pipeline Deep Dive](Pipeline-Deep-Dive.md) for implementation details.

---

## 1. Motivation

The original C++ runtime used a centralized dispatch model with a
`StrategyFamily` enum, a `resolve_family()` switch, and a monolithic
`FastPathModelConfig` struct. This had scaling limitations: adding a new
strategy required editing shared files, the config struct grew with every
modality, and factory logic could not be tested in isolation.

The plugin-registry architecture described below was designed to address
these limitations, and has since been fully implemented.

## 2. Implementation (Current State)

```text
trtmc_create_pipeline_ex(bundle_path)
  -> ReadBundleFile()
  -> extract_json_string("runtime_strategy")
  -> normalize_legacy_strategy()
  -> PipelineRegistry::instance().lookup(strategy)  // returns IPipelinePlugin*
  -> parse_base_config()                             // ~10 universal fields
  -> plugin->create(PipelineContext{...})             // plugin-specific assembly
  -> IPipeline*
```

Key files:

| File | Role |
|------|------|
| `include/trtmc/runtime/pipeline_factory.h` | `PipelineFactory::from_bundle()` declaration |
| `src/runtime/registry/pipeline_factory.cpp` | Thin dispatch: read strategy, lookup plugin, delegate (~124 LOC) |
| `include/trtmc/runtime/pipeline_registry.h` | `PipelineRegistry` singleton, manifest registration macro |
| `src/runtime/registry/pipeline_registry.cpp` | Registry implementation |
| `include/trtmc/runtime/pipeline_plugin.h` | `IPipelinePlugin`, `BaseConfig`, `PipelineContext` |
| `src/runtime/registry/pipeline_plugin.cpp` | `parse_base_config()` |
| `src/runtime/plugins/*.cpp` | Manifest-registered plugin files (25 strategies) |
| `src/runtime/plugins/shared/` | Shared helpers: `plugin_helpers`, `diffusion_helpers`, `audio_helpers` |
| `cmake/trtmc_pipeline_plugins.cmake` | Plugin source/anchor manifest |
| `src/cabi/api/trtmc_c.cpp` | C ABI entry point, calls `PipelineFactory::from_bundle()` |
| `src/runtime/pipelines/*.h/*.cpp` | 14 concrete `IPipeline` implementations |

## 3. Design Details (Implemented)

### 3.1 IPipelinePlugin

```cpp
// include/trtmc/runtime/pipeline_plugin.h
class IPipelinePlugin {
public:
    virtual ~IPipelinePlugin() = default;
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;
};
```

Each plugin receives a `PipelineContext` with the `BundleFile`, `BaseConfig`,
raw JSON text, `hf_python` path, and `bundle_path`. The plugin parses its own
strategy-specific config directly from the raw JSON.

### 3.2 PipelineRegistry

```cpp
// include/trtmc/runtime/pipeline_registry.h
class PipelineRegistry {
public:
    static PipelineRegistry& instance();
    void register_plugin(const std::string& strategy, IPipelinePlugin* plugin);
    IPipelinePlugin* lookup(const std::string& strategy) const;
    std::vector<std::string> registered_strategies() const;
};
```

### 3.3 BaseConfig (Universal Fields)

Instead of a monolithic `FastPathModelConfig`, the factory parses only the ~10
universal fields into `BaseConfig`. Each plugin reads its own strategy-specific
fields from the raw JSON:

```cpp
// include/trtmc/runtime/pipeline_plugin.h
struct BaseConfig {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{1};
    int32_t num_heads{1};
    int32_t num_kv_heads{1};
    int32_t head_dim{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{-1};
    int32_t id_eos{-1};
    std::string runtime_strategy{"decoder_kv_cache"};
    std::string precision{"fp32"};
    bool tokenizer_add_special_tokens{false};
    bool tokenizer_add_special_tokens_present{false};
};
```

### 3.4 Self-Contained Plugins

Each plugin is a single `.cpp` file in `src/runtime/plugins/` that:

- Exposes a manifest-listed registrar function
- Parses strategy-specific config from raw JSON in `create()`
- Extracts bundle sections via `find_section()`
- Loads TRT engines, creates tokenizers and caches
- Returns a fully constructed `IPipeline`

```text
src/runtime/plugins/
  decoder_plugin.cpp          # decoder_kv_cache, decoder_moe
  ssm_plugin.cpp              # ssm_recurrent
  rwkv_plugin.cpp             # rwkv_recurrent
  hybrid_plugin.cpp           # hybrid_mamba_attention
  encoder_plugin.cpp          # encoder_only, embedding, reranking, neural_operator
  vl_plugin.cpp               # vision_language
  segmentation_plugin.cpp     # segmentation, prompted_segmentation
  object_detection_plugin.cpp # object_detection
  whisper_plugin.cpp          # speech_to_text
  bark_plugin.cpp             # text_to_audio_bark
  magpie_plugin.cpp           # text_to_audio_magpie
  speech_plugin.cpp           # speech_to_speech
  omni_plugin.cpp             # omni_multimodal
  t5_plugin.cpp               # text_to_text
  marian_plugin.cpp           # marian_translation
  seq2seq_plugin.cpp          # seq2seq_encoder_decoder
  flux_plugin.cpp             # diffusion_flux
  wan_plugin.cpp              # diffusion_wan, diffusion_pixart
  zimage_plugin.cpp           # diffusion_zimage
  shared/                     # plugin_helpers, diffusion_helpers, audio_helpers
```

## 4. Migration History

All phases have been **completed**.

### Phase 1: Introduce IPipelinePlugin + PipelineRegistry -- DONE

- Defined `IPipelinePlugin` interface in `include/trtmc/runtime/pipeline_plugin.h`.
- Implemented `PipelineRegistry` singleton in `include/trtmc/runtime/pipeline_registry.h`.
- Added `PipelineRegistry` and initial registration macros.

### Phase 2: Decompose FastPathModelConfig -- DONE

- Replaced the monolithic config struct with `BaseConfig` (~10 universal fields).
- Each plugin now parses its strategy-specific config directly from raw JSON.
- `parse_base_config()` in `src/runtime/registry/pipeline_plugin.cpp`.

### Phase 3: Migrate strategies to plugins -- DONE

- All 25 strategies migrated to manifest-registered plugin files in `src/runtime/plugins/`.
- Shared helpers factored into `src/runtime/plugins/shared/`.

### Phase 4: Simplify pipeline_factory.cpp -- DONE

- `PipelineFactory::from_bundle()` is now ~124 LOC: read strategy, normalize legacy strings, lookup plugin, delegate.
- No `resolve_family()` enum, no `StrategyFamily`, no `create_*_pipeline()` functions.

### Phase 5: Plugin manifest registration -- DONE

- Each plugin exposes a registrar function via `REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST`.
- `cmake/trtmc_pipeline_plugins.cmake` drives source inclusion and generated registrar calls.
- External out-of-tree plugins are now architecturally possible.

## 5. C ABI Stability Constraint

The public C ABI defined in `include/trtmc/pipeline.h` **must remain stable throughout the entire migration**:

- `trtmc_create_pipeline()` and `trtmc_create_pipeline_ex()` continue to take a bundle path and return an `IPipeline*`.
- `TrtmcPipelineOptions` struct is not changed.
- The `IPipeline` virtual interface (generate, embed, segment, transcribe, etc.) is not changed.
- All changes are internal to the factory and strategy assembly layer.

Callers of the C ABI will not need any changes at any phase of the migration.

## 6. Testing Strategy for Migration

Each migration phase must maintain the existing test gates:

| Gate | What it validates |
|------|-------------------|
| C++ unit tests (`ctest`) | Bundle parsing, tokenizers, CUDA wrappers, KV cache |
| Python builder tests (`pytest tests/builder/`) | Config parsing, weight mapping, graph ops |
| CCN gate (`tools/check_cyclomatic_complexity.py`) | No function exceeds CCN 10 |
| E2E suite (`pytest tests/test_e2e.py`) | Full pipeline correctness for all 84 model manifests |

Additionally, each new plugin should have:

- **Config parsing unit tests** -- verify that the plugin's `parse_config()` extracts the correct fields and rejects invalid configs.
- **Bundle validation unit tests** -- verify that `validate_bundle()` catches missing sections.
- **Construction unit tests** -- verify that `create_pipeline()` produces a working pipeline (may require TRT harness tests).

## 7. What This Document Is NOT

- This is **not** the current architecture. See [Architecture Overview](Architecture-Overview.md).
- This is **not** an approved migration plan with a schedule. It is a design target.
- This does **not** describe PipelineRouter, PipelineServices, StrategyBuilder, or service-composed runtime patterns. Those concepts do not exist in the codebase and are not part of this target.
- This document now describes the **implemented** plugin-registry architecture. The migration is complete. `PipelineFactory::from_bundle()` is a thin wrapper that delegates to 20 manifest-registered plugins handling 25 strategies across 84 model manifests.
