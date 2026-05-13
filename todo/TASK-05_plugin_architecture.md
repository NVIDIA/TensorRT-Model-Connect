# TASK-05: Plugin Architecture — Registry-Based Pipeline Dispatch

## Branch: `plugin-architecture`

## Motivation

Today, adding a new `runtime_strategy` or performance-tuning an existing pipeline
requires touching the monolithic `pipeline_factory.cpp` (700 LOC) plus two god
structs (`FastPathModelConfig` ~232 fields, `BundleSections` ~40+ fields). This
creates merge conflicts when multiple teams work on different strategies in
parallel and forces every strategy to compile against every other strategy's
config/sections.

**Goal:** Each pipeline is a self-contained plugin. Adding a new strategy or
tuning an existing one touches exactly **one plugin file** — no edits to shared
dispatch code, config structs, or section mappings.

## Prerequisites

- TASK-01 through TASK-04 should be **complete** before starting this work.
  The legacy `IGenerationBackend`/`DecoderStepEngine` path must be gone so we're
  refactoring a clean, uniform `TrtModule`-based codebase.
- All 21 `runtime_strategy` strings must route through the new-pattern pipeline
  classes (no legacy backends).

## Pre-flight Sanity Check (MANDATORY — run before any implementation)

**Before writing any code, the agent MUST run these checks and STOP if any fail.**
Report results to the user and wait for instructions.

### Check 1: No legacy backends in the build
```bash
# Must return ZERO matches. If any .cpp files are still compiled, STOP.
grep -E 'whisper_backend\.cpp|bark_backend\.cpp|magpie_tts_backend\.cpp|speech_backend\.cpp|omni_backend\.cpp' CMakeLists.txt
```

### Check 2: No legacy backend ownership in pipeline classes
```bash
# Must return ZERO matches. If any pipeline still holds a legacy backend ptr, STOP.
grep -E 'unique_ptr<WhisperBackend>|unique_ptr<BarkBackend>|unique_ptr<MagpieTTSBackend>|unique_ptr<SpeechToSpeechBackend>|unique_ptr<OmniBackend>' src/runtime/pipelines/audio_pipeline.h
```

### Check 3: No DecoderStepEngine usage outside legacy headers
```bash
# Must return ZERO matches in .cpp files. Old .h files may still exist (TASK-04 cleanup).
grep -rl 'DecoderStepEngine' src/runtime/pipelines/ src/runtime/pipeline_factory.cpp
```

### Check 4: No DeviceKvCache usage outside legacy headers
```bash
# Must return ZERO matches in .cpp files.
grep -rl 'DeviceKvCache' src/runtime/pipelines/ src/runtime/pipeline_factory.cpp
```

### Check 5: No audio_backend_factory dispatch
```bash
# audio_backend_factory.cpp should either not exist or be empty of factory functions.
# If make_whisper_pipeline_from_bundle / make_bark_pipeline_from_bundle / etc. still
# exist, STOP.
grep -E 'make_whisper_pipeline|make_bark_pipeline|make_magpie_pipeline|make_speech_pipeline' src/runtime/pipelines/audio_backend_factory.cpp src/runtime/pipeline_factory.cpp 2>/dev/null
```

### Check 6: All pipeline classes use TrtModule
```bash
# Every pipeline in audio_pipeline.h should hold TrtModule, not a legacy backend.
# Verify at least Whisper, Bark, Magpie, Speech, Omni all have TrtModule members.
grep -c 'unique_ptr<TrtModule>' src/runtime/pipelines/audio_pipeline.h
# Expected: >= 5 (multiple TrtModule members across the 5 pipeline classes)
```

### What to do if checks fail

**DO NOT proceed with Phase 1.** Instead:
1. Report which checks failed and the exact grep output
2. Identify which TASK (01/02/03/04) is incomplete
3. Ask the user whether to:
   (a) Complete the missing migration first, or
   (b) Proceed with plugin architecture only for the already-migrated strategies

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  trtmc_create_pipeline_ex(bundle_path, options)                  │
│    │                                                             │
│    ▼                                                             │
│  PipelineFactory::from_bundle(path, hf_python)                  │
│    │                                                             │
│    ├─ ReadBundleFile(path)  →  BundleFile                       │
│    ├─ parse_base_config(config_json)  →  BaseConfig             │
│    │     (only shared fields: runtime_strategy, vocab_size,     │
│    │      hidden_size, num_layers, bos/eos, max_cache_length,   │
│    │      tokenizer_add_special_tokens, model_id)               │
│    │                                                             │
│    ├─ PipelineRegistry::lookup(runtime_strategy)                │
│    │     → IPipelinePlugin*                                     │
│    │                                                             │
│    └─ plugin->create(PipelineContext{bundle, base_cfg, hf_py})  │
│          │                                                       │
│          ├─ plugin parses its own config fields from JSON        │
│          ├─ plugin extracts its own sections from BundleFile     │
│          ├─ plugin loads TrtModules, state, tokenizer            │
│          └─ returns unique_ptr<IPipeline>                        │
└──────────────────────────────────────────────────────────────────┘
```

## Design Decisions

### D1: Static registration, not dynamic loading

Plugins are compiled into the binary. Registration uses a global map populated
at static-init time via a `REGISTER_PIPELINE_PLUGIN` macro. No dlopen/dlsym.

**Rationale:** This is a TRT inference runtime — plugin set is fixed at compile
time. Dynamic loading adds complexity (symbol visibility, ABI stability, error
handling) with no benefit. Static registration gives the same decoupling
(no edits to shared code when adding a plugin) without the runtime overhead.

### D2: Plugins own their config parsing

Each plugin receives the raw `config.json` string and extracts only the fields
it needs. No shared config struct beyond `BaseConfig` (the ~10 universal fields).

**Rationale:** This is the key scalability property. The diffusion plugin parses
`num_inference_steps` and `guidance_scale`; the SSM plugin parses `d_inner` and
`conv_kernel`. Neither knows the other exists. Adding a field to one plugin
cannot break another.

### D3: Plugins own their section extraction

Each plugin receives the `BundleFile` and looks up sections by name string
(not via a shared `BundleSections` struct). A thin helper `find_section(bundle, name)`
returns `const vector<char>*` or `nullptr`.

**Rationale:** Same scalability argument as D2. The diffusion plugin looks for
`denoiser_plan` and `vae_decoder_plan`; the text plugin looks for `engine_plan`.
No shared struct to grow.

#### The problem today

`BundleFile` is already generic — it's just `vector<BundleSection>` where each
section is `{name: string, data: vector<char>}`. But immediately after reading
the bundle, we flatten it into a `BundleSections` god struct (40+ pointer fields)
via `find_bundle_sections()` and a compile-time `kSectionMappings[]` table.
Every strategy's fields get populated even though only one strategy will use them.
Adding a section for any strategy means editing the shared struct, the shared
mapping table, and recompiling everything that includes `bundle_helpers.h`.

```
Current flow:
  BundleFile (generic)
    → find_bundle_sections() iterates kSectionMappings[40+]
    → BundleSections god struct (40+ fields, ALL strategies)
    → passed to every create_*_pipeline()
    → each factory reads only 3-5 fields, ignores the rest
```

#### The replacement: direct lookup by name

Two thin helpers (in `bundle_view.h`, ~15 lines total):

```cpp
// Look up a section by exact name. Returns nullptr if not found.
const std::vector<char>* find_section(
    const BundleFile& bundle, const std::string& name);

// Look up all sections matching a prefix (e.g., "text_encoder_" → 0, 1, 2).
// Returns pointers sorted by suffix number.
std::vector<const std::vector<char>*> find_sections_by_prefix(
    const BundleFile& bundle, const std::string& prefix);
```

Implementation is a linear scan of `bundle.sections` (typically 5-15 entries).
Runs once at pipeline creation, not per-inference-step. Negligible cost.

#### How each plugin uses it

Each plugin asks for **exactly the sections it needs** — no shared struct:

```cpp
// decoder_plugin.cpp — only knows about "engine_plan" + tokenizer files
std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
    auto* plan = find_section(ctx.bundle, "engine_plan");
    if (!plan) throw std::runtime_error("Missing engine_plan");
    auto loaded = load_trt_module_from_plan(plan, "decoder");
    auto tokenizer = create_tokenizer_from_bundle(ctx.bundle, ctx.hf_python);
    // ... create TextGenerationPipeline
}

// flux_plugin.cpp — only knows about diffusion sections
std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
    auto* denoiser = find_section(ctx.bundle, "denoiser_plan");
    auto* vae      = find_section(ctx.bundle, "vae_decoder_plan");
    auto* weights  = find_section(ctx.bundle, "preprocessor_weights");
    auto  te_plans = find_sections_by_prefix(ctx.bundle, "text_encoder_");
    // ... load TrtModules, create FluxPipeline
}

// speech_plugin.cpp — only knows about speech sections
std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
    auto* temporal    = find_section(ctx.bundle, "engine_plan");
    auto* mimi_enc    = find_section(ctx.bundle, "mimi_encoder_plan");
    auto* mimi_dec    = find_section(ctx.bundle, "mimi_decoder_plan");
    auto  depth_plans = find_sections_by_prefix(ctx.bundle, "depth_engine_plan_");
    auto* audio_embed = find_section(ctx.bundle, "audio_embeddings");
    // ... load TrtModules, create SpeechPipeline
}
```

#### Adding a new strategy with custom sections

```cpp
// my_new_plugin.cpp — zero edits to any shared file
std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
    auto* enc   = find_section(ctx.bundle, "my_encoder_plan");   // new name
    auto* dec   = find_section(ctx.bundle, "my_decoder_plan");   // new name
    auto* vocab = find_section(ctx.bundle, "my_custom_vocab");   // new name
    // ...
}
REGISTER_PIPELINE_PLUGIN(MyNewPlugin);
```

The `BundleFile` already has the data (written by the Python builder). The
plugin just asks for it by name. No `BundleSections` field to add, no
`kSectionMappings[]` entry, no `bundle_helpers.h` recompile.

#### Shared utilities adapt too

`extract_tokenizer_from_bundle` and `extract_clip_tokenizer_from_bundle` today
take `const BundleSections&` and read 7-8 specific fields. After D3, they take
`const BundleFile&` and call `find_section()` internally:

```cpp
TokenizerResult create_tokenizer_from_bundle(
    const BundleFile& bundle, const std::string& hf_python, bool add_special)
{
    auto* tok_json   = find_section(bundle, "tokenizer.json");
    auto* tok_config = find_section(bundle, "tokenizer_config.json");
    auto* vocab      = find_section(bundle, "vocab.json");
    auto* merges     = find_section(bundle, "merges.txt");
    auto* special    = find_section(bundle, "special_tokens_map.json");
    auto* model      = find_section(bundle, "tokenizer.model");
    auto* preproc    = find_section(bundle, "preprocessor_config.json");
    // write to tmpdir, create HfPythonTokenizer — same logic as today
}
```

#### What gets deleted

- `BundleSections` struct (40+ fields) — deleted entirely
- `kSectionMappings[]` table (40+ entries) — deleted entirely
- `find_bundle_sections()` function — replaced by `find_section()` / `find_sections_by_prefix()`
- `assign_mapped_section()`, `assign_depth_engine_plan_section()` — deleted

### D4: Shared utilities stay shared

`load_trt_module_from_plan()`, `create_tokenizer_from_bundle()`, `KvCache`,
`RecurrentState`, `TrtModule`, `CudaStream` — these are **not** plugin-specific.
They live in a shared utility layer that all plugins can use.

### D5: IPipeline interface is unchanged

The public `IPipeline` interface and C ABI are not touched. This is a
purely internal refactoring.

### D6: One plugin per file, one strategy per plugin (with grouping allowed)

Most plugins handle exactly one `runtime_strategy`. Where strategies share
95%+ logic (e.g., `decoder_kv_cache` and `decoder_moe`), a single plugin can
register for multiple strategies. But the default is 1:1.

## Concrete Interfaces

### `IPipelinePlugin` (new)

```cpp
// include/trtmc/runtime/pipeline_plugin.h

#pragma once
#include "trtmc/pipeline.h"
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct BundleFile;  // forward

// Minimal shared config — only fields needed by ALL strategies.
struct BaseConfig {
    std::string runtime_strategy;
    std::string model_id;
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{0};
    int32_t num_heads{0};
    int32_t num_kv_heads{0};
    int32_t head_dim{0};
    int32_t max_cache_length{256};
    int32_t id_bos{1};
    int32_t id_eos{2};
    bool tokenizer_add_special_tokens{true};
    std::string config_json;         // raw JSON — plugins parse their own fields
};

// Everything a plugin needs to create a pipeline.
struct PipelineContext {
    const BundleFile& bundle;        // raw bundle (sections accessible by name)
    const BaseConfig& config;        // shared base config
    std::string hf_python;           // path to python for tokenizer
    std::string bundle_path;         // for plugins that need the path (e.g. ZImage)
};

class IPipelinePlugin {
public:
    virtual ~IPipelinePlugin() = default;

    // Human-readable name for logging
    virtual const char* name() const = 0;

    // Which runtime_strategy strings this plugin handles.
    // Called once at registration time.
    virtual std::vector<std::string> strategies() const = 0;

    // Create a pipeline from the bundle.
    // Plugin is responsible for:
    //   - Parsing strategy-specific config from ctx.config.config_json
    //   - Extracting sections from ctx.bundle by name
    //   - Loading TrtModules, state objects, tokenizers
    //   - Returning a fully constructed IPipeline
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;
};

} // namespace trtmc
```

### `PipelineRegistry` (new)

```cpp
// include/trtmc/runtime/pipeline_registry.h

#pragma once
#include "trtmc/runtime/pipeline_plugin.h"
#include <string>
#include <unordered_map>

namespace trtmc {

class PipelineRegistry {
public:
    static PipelineRegistry& instance();

    // Register a plugin for all its declared strategies.
    void register_plugin(std::unique_ptr<IPipelinePlugin> plugin);

    // Look up a plugin by runtime_strategy string.
    // Returns nullptr if no plugin registered for this strategy.
    IPipelinePlugin* lookup(const std::string& strategy) const;

    // List all registered strategy strings (for error messages / inspect).
    std::vector<std::string> registered_strategies() const;

private:
    PipelineRegistry() = default;
    std::vector<std::unique_ptr<IPipelinePlugin>> plugins_;
    std::unordered_map<std::string, IPipelinePlugin*> strategy_map_;
};

// Macro for static registration. Place in .cpp file at file scope:
//   REGISTER_PIPELINE_PLUGIN(MyPlugin);
// expands to a static initializer that creates a MyPlugin and registers it.
#define REGISTER_PIPELINE_PLUGIN(PluginClass) \
    namespace { \
    static const bool kRegistered_##PluginClass = [] { \
        ::trtmc::PipelineRegistry::instance().register_plugin( \
            std::make_unique<PluginClass>()); \
        return true; \
    }(); \
    }

} // namespace trtmc
```

### `BundleFile` section lookup helper (new)

```cpp
// Add to bundle_format.h or a new bundle_view.h

// Look up a section by name. Returns nullptr if not found.
const std::vector<char>* find_section(const BundleFile& bundle, const std::string& name);

// Look up all sections matching a prefix (e.g., "text_encoder_" → 0,1,2...).
// Returns pointers in sorted order by suffix number.
std::vector<const std::vector<char>*> find_sections_by_prefix(
    const BundleFile& bundle, const std::string& prefix);
```

## Plugin Catalog

Each row = one plugin file. "Strategies" = the `runtime_strategy` strings it
registers for.

| Plugin file | Strategies | Pipeline class | Sections used | Config fields (beyond base) |
|---|---|---|---|---|
| `decoder_plugin.cpp` | `decoder_kv_cache`, `decoder_moe` | `TextGenerationPipeline` | `engine_plan`, tokenizer files | `attention_size` |
| `ssm_plugin.cpp` | `ssm_recurrent` | `RecurrentPipeline` | `engine_plan`, tokenizer files | `d_inner`, `state_size`, `conv_kernel` |
| `rwkv_plugin.cpp` | `rwkv_recurrent` | `RecurrentPipeline` | `engine_plan`, tokenizer files | (none beyond base) |
| `hybrid_plugin.cpp` | `hybrid_mamba_attention` | `RecurrentPipeline` | `engine_plan`, tokenizer files | `num_mamba_layers`, `num_attention_layers`, `d_inner`, `mamba_d_state`, `mamba_d_conv`, `mamba_nheads`, `mamba_head_dim`, `conv_dim` |
| `encoder_plugin.cpp` | `encoder_only`, `embedding`, `reranking`, `neural_operator` | `EncoderPipeline` | `engine_plan`, tokenizer files | `type_vocab_size`, `embedding_dim`, `operator_type`, etc. |
| `segmentation_plugin.cpp` | `segmentation`, `prompted_segmentation` | `SegmentPipeline`, `SamPipeline` | `engine_plan`, `vision_engine_plan` | `num_classes`, `input_image_h/w`, SAM fields |
| `object_detection_plugin.cpp` | `object_detection` | `EncoderPipeline` | `engine_plan` | `det_num_classes`, `det_input_h/w`, `det_conf/nms_threshold` |
| `vl_plugin.cpp` | `vision_language` | `VLPipeline` | `engine_plan`, `vision_engine_plan`, `preprocessor_config.json`, tokenizer files | `image_token_id`, `vision_output_dim`, `has_vision_engine`, `embed_input`, VL preprocess fields |
| `whisper_plugin.cpp` | `speech_to_text` | `WhisperPipeline` | `engine_plan`, `vision_engine_plan` or `coarse_engine_plan`, `mel_filterbank`, tokenizer files | `num_mel_bins`, `mel_*`, `encoder_layers`, `decoder_layers`, `eot_token_id`, `decoder_start_token_ids` |
| `bark_plugin.cpp` | `text_to_audio` (is_magpie_tts=false) | `BarkPipeline` | `engine_plan`, `coarse_engine_plan`, `fine_engine_plan`, `codec_engine_plan`, `semantic/coarse/fine_embed`, tokenizer files | `sample_rate`, `*_vocab_size`, `n_*_codebooks`, `coarse_*`, `fine_*`, `codec_*` |
| `magpie_plugin.cpp` | `text_to_audio_magpie` | `MagpiePipeline` | `engine_plan`, `vision_engine_plan`, `codec_engine_plan`, `magpie_*`, IPA tokenizer files | `magpie_*` fields |
| `speech_plugin.cpp` | `speech_to_speech` | `SpeechPipeline` | `engine_plan`, `depth_engine_plan*`, `mimi_*`, embedding sections | `sample_rate`, `num_codebooks`, `depth_*`, `speech_*` |
| `omni_plugin.cpp` | `omni_multimodal` | `OmniPipeline` | `engine_plan`, `talker_engine_plan`, `code2wav_engine_plan`, tokenizer files | `omni_*` fields, VL fields |
| `flux_plugin.cpp` | `diffusion_flux` | `FluxPipeline` | `text_encoder_*_plan`, `denoiser_plan`, `vae_decoder_plan`, `preprocessor_weights`, `clip_*` tokenizer files, main tokenizer files | diffusion fields + FLUX-specific |
| `wan_plugin.cpp` | `diffusion_wan` | `WanPipeline` | `text_encoder_*_plan`, `denoiser_plan`, `vae_decoder_plan`, `preprocessor_weights`, tokenizer files | diffusion fields + Wan-specific |
| `zimage_plugin.cpp` | `diffusion_zimage` | `ZImagePipeline` | `text_encoder_*_plan`, `denoiser_plan`, `vae_decoder_plan`, `preprocessor_weights`, tokenizer files | diffusion fields + Z-Image-specific |

### Strategy splitting: no sub-dispatch allowed

**Principle:** One `runtime_strategy` string = one plugin = one pipeline class.
No `can_handle()`, no secondary config field inspection, no ambiguity. The
registry is a simple 1:1 map from strategy string to plugin.

Two strategies currently violate this and must be split:

#### 1. `text_to_audio` → split into `text_to_audio_bark` + `text_to_audio_magpie`

**Today:** Both `bark.py` and `magpie_tts.py` Python family plugins write
`runtime_strategy = "text_to_audio"`. The C++ runtime sub-dispatches on
`cfg.is_magpie_tts` (parsed from `magpie_tts: 1` in config.json).

**After:** Each plugin writes a distinct strategy string:

```python
# tensorrt_model_connect/tensorrt_model_connect/families/bark.py
class BarkPlugin(FamilyPlugin):
    runtime_strategy = "text_to_audio_bark"       # was "text_to_audio"

# tensorrt_model_connect/tensorrt_model_connect/families/magpie_tts.py
class MagpieTTSPlugin(FamilyPlugin):
    runtime_strategy = "text_to_audio_magpie"     # was "text_to_audio"
```

#### 2. `diffusion` → split into `diffusion_flux` + `diffusion_wan` + `diffusion_zimage`

**Today:** `flux.py`, `wan_t2v.py`, `z_image.py`, and `pixart.py` all write
`runtime_strategy = "diffusion"`. The C++ runtime sub-dispatches on
`cfg.diffusion_backend_type` (a secondary config field like `"flux_2d"`,
`"wan_3d"`, `"z_image_2d"`).

**After:** Each plugin writes a distinct strategy string:

```python
# tensorrt_model_connect/tensorrt_model_connect/families/flux.py
class FluxPlugin(FamilyPlugin):
    runtime_strategy = "diffusion_flux"           # was "diffusion"

# tensorrt_model_connect/tensorrt_model_connect/families/wan_t2v.py
class WanPlugin(FamilyPlugin):
    runtime_strategy = "diffusion_wan"            # was "diffusion"

# tensorrt_model_connect/tensorrt_model_connect/families/z_image.py
class ZImagePlugin(FamilyPlugin):
    runtime_strategy = "diffusion_zimage"         # was "diffusion"

# tensorrt_model_connect/tensorrt_model_connect/families/pixart.py
class PixArtPlugin(FamilyPlugin):
    runtime_strategy = "diffusion_pixart"         # was "diffusion"
```

#### Files that must be updated for the split

**Python builder (strategy strings):**
- `tensorrt_model_connect/tensorrt_model_connect/families/bark.py` — `runtime_strategy = "text_to_audio_bark"`
- `tensorrt_model_connect/tensorrt_model_connect/families/magpie_tts.py` — `runtime_strategy = "text_to_audio_magpie"`
- `tensorrt_model_connect/tensorrt_model_connect/families/flux.py` — `runtime_strategy = "diffusion_flux"`
- `tensorrt_model_connect/tensorrt_model_connect/families/wan_t2v.py` — `runtime_strategy = "diffusion_wan"`
- `tensorrt_model_connect/tensorrt_model_connect/families/z_image.py` — `runtime_strategy = "diffusion_zimage"`
- `tensorrt_model_connect/tensorrt_model_connect/families/pixart.py` — `runtime_strategy = "diffusion_pixart"`

**E2E harness (strategy→task mapping):**
- `tests/e2e_harness/contracts.py` — `RUNTIME_TO_TASK_STRATEGY`: add new keys
  ```python
  "text_to_audio_bark": "text_to_audio",
  "text_to_audio_magpie": "text_to_audio",
  "diffusion_flux": "diffusion_media_generation",
  "diffusion_wan": "diffusion_media_generation",
  "diffusion_zimage": "diffusion_media_generation",
  "diffusion_pixart": "diffusion_media_generation",
  ```
  Remove old `"text_to_audio"` and `"diffusion"` entries.
- `tests/e2e_harness/manifest_loader.py` — `_DEFAULT_STAGES`: add entries for
  new strategy strings (or map them to the same defaults as the old ones).
- `tests/e2e/models/*.json` — update `runtime_strategy` in affected manifests
  (bark, magpie, flux, wan, z-image, pixart models).

**C++ runtime (dispatch):**
- `src/runtime/pipeline_factory.cpp` — update `resolve_family()` map and
  `create_audio_pipeline()` / `create_diffusion_pipeline()` to use new strings.
  Remove `is_magpie_tts` sub-check and `is_flux_type`/`is_zimage_type` helpers.
- `src/cabi/config/fast_path_config.cpp` — `kHandlers` table: add new strategy
  strings, remove old `"text_to_audio"` and `"diffusion"` entries.

**Python debug runner:**
- `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` — handle new strategy strings.

**Old bundles:** Must be rebuilt with new strategy strings. No backward compat
shims — old bundles with `"text_to_audio"` or `"diffusion"` will fail with
"No plugin for strategy" error, which is the correct behavior (rebuild required).

#### Complete strategy list after splitting

| `runtime_strategy` | Plugin | Pipeline class |
|---|---|---|
| `decoder_kv_cache` | `decoder_plugin.cpp` | `TextGenerationPipeline` |
| `decoder_moe` | `decoder_plugin.cpp` | `TextGenerationPipeline` |
| `ssm_recurrent` | `ssm_plugin.cpp` | `RecurrentPipeline` |
| `rwkv_recurrent` | `rwkv_plugin.cpp` | `RecurrentPipeline` |
| `hybrid_mamba_attention` | `hybrid_plugin.cpp` | `RecurrentPipeline` |
| `encoder_only` | `encoder_plugin.cpp` | `EncoderPipeline` |
| `embedding` | `encoder_plugin.cpp` | `EncoderPipeline` |
| `reranking` | `encoder_plugin.cpp` | `EncoderPipeline` |
| `neural_operator` | `encoder_plugin.cpp` | `EncoderPipeline` |
| `segmentation` | `segmentation_plugin.cpp` | `SegmentPipeline` |
| `prompted_segmentation` | `segmentation_plugin.cpp` | `SamPipeline` |
| `object_detection` | `object_detection_plugin.cpp` | `EncoderPipeline` |
| `vision_language` | `vl_plugin.cpp` | `VLPipeline` |
| `speech_to_text` | `whisper_plugin.cpp` | `WhisperPipeline` |
| `text_to_audio_bark` | `bark_plugin.cpp` | `BarkPipeline` |
| `text_to_audio_magpie` | `magpie_plugin.cpp` | `MagpiePipeline` |
| `speech_to_speech` | `speech_plugin.cpp` | `SpeechPipeline` |
| `omni_multimodal` | `omni_plugin.cpp` | `OmniPipeline` |
| `diffusion_flux` | `flux_plugin.cpp` | `FluxPipeline` |
| `diffusion_wan` | `wan_plugin.cpp` | `WanPipeline` |
| `diffusion_zimage` | `zimage_plugin.cpp` | `ZImagePipeline` |
| `diffusion_pixart` | `pixart_plugin.cpp` | `PixArtPipeline` |

23 strategies, 16 plugin files, 0 sub-dispatch.

## Implementation Phases

### Phase 1: Scaffolding (non-breaking)

**Goal:** Introduce `IPipelinePlugin`, `PipelineRegistry`, `BaseConfig`,
`find_section()`, and the `REGISTER_PIPELINE_PLUGIN` macro — but don't move
any existing logic yet. The old `pipeline_factory.cpp` dispatch still works.

**Files created:**
- `include/trtmc/runtime/pipeline_plugin.h` — `IPipelinePlugin`, `BaseConfig`, `PipelineContext`
- `include/trtmc/runtime/pipeline_registry.h` — `PipelineRegistry`, `REGISTER_PIPELINE_PLUGIN`
- `src/runtime/pipeline_registry.cpp` — `PipelineRegistry` implementation
- `src/bundle/bundle_view.h` / `.cpp` — `find_section()`, `find_sections_by_prefix()`

**Files modified:**
- `CMakeLists.txt` — add new sources

**Verification:**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
```

### Phase 2: Split ambiguous strategy strings

**Goal:** Eliminate all sub-dispatch. Each `runtime_strategy` maps to exactly
one pipeline class. No `can_handle()`, no secondary field inspection.

**Python builder changes:**
- `tensorrt_model_connect/tensorrt_model_connect/families/bark.py` — `runtime_strategy = "text_to_audio_bark"`
- `tensorrt_model_connect/tensorrt_model_connect/families/magpie_tts.py` — `runtime_strategy = "text_to_audio_magpie"`
- `tensorrt_model_connect/tensorrt_model_connect/families/flux.py` — `runtime_strategy = "diffusion_flux"`
- `tensorrt_model_connect/tensorrt_model_connect/families/wan_t2v.py` — `runtime_strategy = "diffusion_wan"`
- `tensorrt_model_connect/tensorrt_model_connect/families/z_image.py` — `runtime_strategy = "diffusion_zimage"`
- `tensorrt_model_connect/tensorrt_model_connect/families/pixart.py` — `runtime_strategy = "diffusion_pixart"`

**C++ runtime changes:**
- `src/runtime/pipeline_factory.cpp` — add `normalize_legacy_strategy()` that
  rewrites old `"text_to_audio"` → `"text_to_audio_bark"/"text_to_audio_magpie"`
  and `"diffusion"` → `"diffusion_flux"/"diffusion_wan"/"diffusion_zimage"` using
  secondary config fields. Update `resolve_family()` map with new strings.
  Remove `is_flux_type()`/`is_zimage_type()` helpers and `is_magpie_tts` sub-check.
- `src/cabi/config/fast_path_config.cpp` — `kHandlers` table: add new strategy
  strings pointing to same parsers. Keep old strings as aliases.

**E2E harness changes:**
- `tests/e2e_harness/contracts.py` — `RUNTIME_TO_TASK_STRATEGY`: add new keys
- `tests/e2e_harness/manifest_loader.py` — `_DEFAULT_STAGES`: add new keys
- `tests/e2e/models/*.json` — update `runtime_strategy` in affected manifests

**Debug runner:**
- `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` — handle new strategy strings

**Verification:**
```bash
# Python tests (builder + tools)
python -m pytest tests/builder/ tests/tools/ -v

# C++ build + tests
cmake --build build -j && ctest --test-dir build --output-on-failure

# E2E smoke for each split strategy (requires GPU):
pytest tests/test_e2e.py::test_e2e[bark-small] \
       tests/test_e2e.py::test_e2e[flux-schnell] \
       tests/test_e2e.py::test_e2e[wan21-t2v-1.3b] \
  -v --engine-dir ... --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

### Phase 3: Extract first plugin (proof-of-concept) — `decoder_plugin.cpp`

**Goal:** Migrate `decoder_kv_cache` + `decoder_moe` to the plugin pattern
as a proof-of-concept. Both the old path and new path work — `pipeline_factory.cpp`
tries registry first, falls back to old dispatch.

**Files created:**
- `src/runtime/plugins/decoder_plugin.cpp`

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — add registry-first lookup in `from_bundle()`
- `CMakeLists.txt` — add plugin source

**Key implementation detail:** The `from_bundle()` entry point becomes:

```cpp
auto plugin = PipelineRegistry::instance().lookup(cfg.runtime_strategy);
if (plugin) {
    PipelineContext ctx{bundle, base_cfg, hf_python, bundle_path};
    return plugin->create(ctx);
}
// fallback to old dispatch_pipeline()
return dispatch_pipeline(...);
```

**Verification:**
```bash
# Must pass — decoder_kv_cache and decoder_moe now served by plugin
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] tests/test_e2e.py::test_e2e[mixtral-small] -v \
  --engine-dir ... --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

### Phase 4: Extract all text-family plugins

**Files created:**
- `src/runtime/plugins/ssm_plugin.cpp`
- `src/runtime/plugins/rwkv_plugin.cpp`
- `src/runtime/plugins/hybrid_plugin.cpp`

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — remove `create_text_pipeline()` and helpers
- `CMakeLists.txt`

**Verification:**
```bash
pytest tests/test_e2e.py -v --e2e-task-strategy text_generation_causal \
  --engine-dir ... --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

### Phase 5: Extract encoder + vision plugins

**Files created:**
- `src/runtime/plugins/encoder_plugin.cpp`
- `src/runtime/plugins/segmentation_plugin.cpp`
- `src/runtime/plugins/object_detection_plugin.cpp`
- `src/runtime/plugins/vl_plugin.cpp`

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — remove `create_encoder_pipeline()`, `create_vision_pipeline()`
- `CMakeLists.txt`

### Phase 6: Extract diffusion plugins

**Files created:**
- `src/runtime/plugins/flux_plugin.cpp`
- `src/runtime/plugins/wan_plugin.cpp`
- `src/runtime/plugins/zimage_plugin.cpp`

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — remove `create_diffusion_pipeline()` and all diffusion helpers
- `CMakeLists.txt`

**Note:** Shared diffusion utilities (`make_diffusion_config`, `parse_preprocessor_weights`,
`DiffusionConfig`, `PreprocessorWeights`) move to a shared diffusion utility header
that all three plugins include. This is intentional — shared utilities ≠ shared dispatch.

### Phase 7: Extract audio plugins

**Files created:**
- `src/runtime/plugins/whisper_plugin.cpp`
- `src/runtime/plugins/bark_plugin.cpp`
- `src/runtime/plugins/magpie_plugin.cpp`
- `src/runtime/plugins/speech_plugin.cpp`
- `src/runtime/plugins/omni_plugin.cpp`

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — remove `create_audio_pipeline()` and `create_omni_pipeline()`
- `CMakeLists.txt`

### Phase 8: Delete god structs + shrink factory

**Goal:** After all plugins are extracted, `pipeline_factory.cpp` shrinks to ~50 lines:

```cpp
std::unique_ptr<IPipeline> PipelineFactory::from_bundle(
    const std::string& bundle_path, const std::string& hf_python)
{
    auto bundle = ReadBundleFile(bundle_path);
    auto base_cfg = parse_base_config(bundle);  // ~10 fields only
    auto* plugin = PipelineRegistry::instance().lookup(base_cfg.runtime_strategy);
    if (!plugin)
        throw std::runtime_error("No plugin for strategy: " + base_cfg.runtime_strategy);
    PipelineContext ctx{bundle, base_cfg, hf_python, bundle_path};
    return plugin->create(ctx);
}
```

**Files deleted:**
- `src/cabi/config/fast_path_config.h` — replaced by `BaseConfig` + per-plugin parsing
- `src/cabi/config/fast_path_config.cpp` — same
- `src/cabi/bundle/bundle_helpers.h` — `BundleSections` struct deleted, `find_bundle_sections()` deleted.
  `extract_tokenizer_from_bundle()` and `load_mel_filterbank()` move to shared utilities.

**Files modified:**
- `src/runtime/pipeline_factory.cpp` — shrinks to ~50 lines (above)
- `CMakeLists.txt` — remove deleted sources, update include paths

**Note:** `parse_base_config()` replaces `parse_fast_path_config()`. It only
extracts the ~10 universal fields. Each plugin has its own `parse_*_config(json)`
that reads the raw JSON string.

### Phase 9: Tests + documentation

- Add `tests/cpp/test_pipeline_registry.cpp` — unit test for registry mechanics:
  register, lookup, unknown strategy error, legacy normalization
- Update `CLAUDE.md` source layout and "Adding a new model family" section
- Update `website/docs/wiki/` if applicable

## File Layout After Migration

```
include/trtmc/
  pipeline.h                          # IPipeline (unchanged)
  runtime/
    pipeline_factory.h                # PipelineFactory (unchanged interface, ~50 LOC impl)
    pipeline_plugin.h                 # IPipelinePlugin, BaseConfig, PipelineContext
    pipeline_registry.h               # PipelineRegistry, REGISTER_PIPELINE_PLUGIN
    trt_module.h                      # TrtModule (unchanged)
    kv_cache.h                        # KvCache (unchanged)
    recurrent_state.h                 # RecurrentState (unchanged)
    scheduler.h                       # IScheduler (unchanged)

src/runtime/
  pipeline_factory.cpp                # ~50 lines: load → parse base → registry lookup → create
  pipeline_registry.cpp               # Registry singleton + register/lookup

  plugins/                            # ← NEW: one file per strategy (or group)
    decoder_plugin.cpp                # decoder_kv_cache, decoder_moe
    ssm_plugin.cpp                    # ssm_recurrent
    rwkv_plugin.cpp                   # rwkv_recurrent
    hybrid_plugin.cpp                 # hybrid_mamba_attention
    encoder_plugin.cpp                # encoder_only, embedding, reranking, neural_operator
    segmentation_plugin.cpp           # segmentation, prompted_segmentation
    object_detection_plugin.cpp       # object_detection
    vl_plugin.cpp                     # vision_language
    whisper_plugin.cpp                # speech_to_text
    bark_plugin.cpp                   # text_to_audio (bark)
    magpie_plugin.cpp                 # text_to_audio (magpie)
    speech_plugin.cpp                 # speech_to_speech
    omni_plugin.cpp                   # omni_multimodal
    flux_plugin.cpp                   # diffusion (flux)
    wan_plugin.cpp                    # diffusion (wan)
    zimage_plugin.cpp                 # diffusion (z_image)

  plugins/shared/                     # ← Shared utilities used by multiple plugins
    plugin_helpers.h                  # load_trt_module_from_plan, create_tokenizer_from_bundle
    diffusion_helpers.h               # DiffusionConfig, PreprocessorWeights, shared diffusion math
    audio_helpers.h                   # Mel extraction, audio resampling

  pipelines/                          # Pipeline implementations (unchanged)
    text_generation_pipeline.h/cpp
    recurrent_pipeline.h/cpp
    vl_pipeline.h/cpp
    encoder_pipeline.h/cpp
    diffusion_pipeline.h              # FluxPipeline, WanPipeline, ZImagePipeline
    audio_pipeline.h/cpp              # WhisperPipeline, BarkPipeline, etc.

src/bundle/
  bundle_format.h/cpp                 # BundleFile, ReadBundleFile (unchanged)
  bundle_view.h/cpp                   # find_section(), find_sections_by_prefix() (NEW)
```

## Adding a New Strategy After Migration

To add a brand new `runtime_strategy` (e.g., `graph_neural_network`):

```bash
# 1. Create the plugin (one file)
cat > src/runtime/plugins/gnn_plugin.cpp << 'CPP'
#include "trtmc/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/pipelines/encoder_pipeline.h"  // or a new GnnPipeline

namespace trtmc {

class GnnPlugin : public IPipelinePlugin {
public:
    const char* name() const override { return "GnnPlugin"; }
    std::vector<std::string> strategies() const override { return {"graph_neural_network"}; }

    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        // Parse GNN-specific config from raw JSON
        auto gnn_cfg = parse_gnn_config(ctx.config.config_json);

        // Load engine
        auto* plan = find_section(ctx.bundle, "engine_plan");
        auto loaded = load_trt_module_from_plan(plan, "gnn_engine");

        // Create pipeline
        return std::make_unique<EncoderPipeline>(
            std::move(loaded.module), "graph_neural_network",
            nullptr, ctx.config.model_id);
    }

private:
    struct GnnConfig { int num_message_passing_layers; /* ... */ };
    GnnConfig parse_gnn_config(const std::string& json) { /* ... */ }
};

REGISTER_PIPELINE_PLUGIN(GnnPlugin);

} // namespace trtmc
CPP

# 2. Add to CMakeLists.txt
#    src/runtime/plugins/gnn_plugin.cpp

# 3. Build + test
cmake --build build -j
pytest tests/test_e2e.py::test_e2e[my-gnn-model] -v ...
```

**Zero edits** to `pipeline_factory.cpp`, `pipeline_registry.cpp`,
`BaseConfig`, or any other plugin file.

## Performance Tuning Isolation

The plugin pattern also isolates performance work. Example: tuning the
decoder pipeline's KV cache memory layout.

**Before (monolithic):**
- Touch `pipeline_factory.cpp` (create_decoder_pipeline)
- Touch `FastPathModelConfig` (add config knobs)
- Risk breaking audio/diffusion/VL code paths

**After (plugin):**
- Touch `src/runtime/plugins/decoder_plugin.cpp` only
- Add local config parsing for new knobs
- No merge conflict risk with other strategy teams

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Static initializer ordering across TUs | Registry is a Meyer's singleton (`static local`). Plugins register in indeterminate order, but that's fine — all registration completes before `main()`. |
| Linker may discard unreferenced plugin TUs | CMake `OBJECT` library or `--whole-archive` linker flag for the plugins directory. Alternatively, explicit `#include` of each plugin in a `plugins_init.cpp` file. |
| Old bundles with `"diffusion"` or `"text_to_audio"` strategy | Old bundles must be rebuilt. Registry returns clear error: "No plugin for strategy: diffusion". |
| God struct removal is a large diff | Phase 8 is the only breaking phase. All prior phases are additive. Can be split into sub-PRs (one per deleted struct). |
| Config JSON parsing duplicated across plugins | Extract a lightweight `JsonReader` utility (already have `json_helpers.h`). Each plugin calls `extract_json_int(json, "field_name")` — same helpers, different fields. |

## Verification Strategy

Each phase has its own verification gate (listed above). The full regression
gate after Phase 8:

```bash
# Tier 1: Unit tests
ctest --test-dir build --output-on-failure
python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
python -m pytest tests/tools/ -v

# Tier 2: Registry unit test
ctest --test-dir build -R test_pipeline_registry --output-on-failure

# Tier 3: E2E smoke (one model per strategy family)
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] \
       tests/test_e2e.py::test_e2e[mamba-130m] \
       tests/test_e2e.py::test_e2e[bert-base-uncased] \
       tests/test_e2e.py::test_e2e[qwen25vl-3b] \
       tests/test_e2e.py::test_e2e[whisper-tiny] \
       tests/test_e2e.py::test_e2e[bark-small] \
       tests/test_e2e.py::test_e2e[flux-schnell] \
       tests/test_e2e.py::test_e2e[segformer-b0-ade] \
  -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# Tier 4: Full E2E (all 50 models)
pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# CCN gate
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

## Coverage-Based Testing Policy

The plugin architecture creates a clear isolation boundary: changes inside a
plugin file **cannot** affect other plugins. This enables a coverage-based
testing policy where CI selects the minimal test set based on which files
changed, and the full suite runs only in nightly.

### Isolation guarantees

| Layer | Isolation property |
|---|---|
| Plugin `.cpp` file | Self-contained. Own config parsing, own section extraction, own TrtModule loading. No shared state with other plugins. |
| Pipeline class (e.g. `FluxPipeline`) | Used by one or a few plugins. Change here = test all plugins that construct this class. |
| Shared utility (`KvCache`, `TrtModule`, `plugin_helpers.h`) | Used by many/all plugins. Change here = test all affected plugins. |
| Infrastructure (`PipelineRegistry`, `bundle_view`, `pipeline_factory`) | Used by everything. Change here = full suite. Rarely changes. |

### File → test mapping

This mapping is the source of truth for CI test selection. It maps each file
(or directory) to the minimal set of E2E models that must pass.

```python
# ci/test_coverage_map.py (or equivalent CI config)
#
# Key: file glob pattern
# Value: list of E2E test IDs (pytest -k compatible)

COVERAGE_MAP = {
    # ── Plugin files: test only that plugin's models ──
    "src/runtime/plugins/decoder_plugin.cpp": [
        "qwen3-0.6b", "tinyllama-1.1b", "llama-3.1-8b",
    ],
    "src/runtime/plugins/ssm_plugin.cpp": [
        "mamba-130m",
    ],
    "src/runtime/plugins/rwkv_plugin.cpp": [
        "rwkv-7-0.1b",
    ],
    "src/runtime/plugins/hybrid_plugin.cpp": [
        "nemotron-h-8b",
    ],
    "src/runtime/plugins/encoder_plugin.cpp": [
        "bert-base-uncased", "eagle-embed",
    ],
    "src/runtime/plugins/segmentation_plugin.cpp": [
        "segformer-b0-ade", "sam-vit-base",
    ],
    "src/runtime/plugins/object_detection_plugin.cpp": [
        # add when model exists
    ],
    "src/runtime/plugins/vl_plugin.cpp": [
        "qwen25vl-3b", "qwen3vl-2b", "internvl3-1b",
    ],
    "src/runtime/plugins/whisper_plugin.cpp": [
        "whisper-tiny",
    ],
    "src/runtime/plugins/bark_plugin.cpp": [
        "bark-small",
    ],
    "src/runtime/plugins/magpie_plugin.cpp": [
        "magpie-tts-357m",
    ],
    "src/runtime/plugins/speech_plugin.cpp": [
        "personaplex-7b",
    ],
    "src/runtime/plugins/omni_plugin.cpp": [
        # add when omni model manifest exists
    ],
    "src/runtime/plugins/flux_plugin.cpp": [
        "flux-schnell",
    ],
    "src/runtime/plugins/wan_plugin.cpp": [
        "wan21-t2v-1.3b",
    ],
    "src/runtime/plugins/zimage_plugin.cpp": [
        "z-image-turbo",
    ],

    # ── Pipeline classes: test all plugins using that class ──
    "src/runtime/pipelines/text_generation_pipeline.*": [
        "qwen3-0.6b", "tinyllama-1.1b",
    ],
    "src/runtime/pipelines/recurrent_pipeline.*": [
        "mamba-130m", "rwkv-7-0.1b", "nemotron-h-8b",
    ],
    "src/runtime/pipelines/vl_pipeline.*": [
        "qwen25vl-3b", "qwen3vl-2b", "internvl3-1b",
    ],
    "src/runtime/pipelines/encoder_pipeline.*": [
        "bert-base-uncased", "segformer-b0-ade",
    ],
    "src/runtime/pipelines/diffusion_pipeline.*": [
        "flux-schnell", "wan21-t2v-1.3b", "z-image-turbo",
    ],
    "src/runtime/pipelines/audio_pipeline.*": [
        "whisper-tiny", "bark-small", "magpie-tts-357m", "personaplex-7b",
    ],

    # ── Shared utilities: test all plugins that depend on them ──
    "include/trtmc/runtime/kv_cache.*": [
        "qwen3-0.6b", "qwen25vl-3b", "whisper-tiny", "bark-small",
        "flux-schnell",  # any attention-based pipeline
    ],
    "src/runtime/trt/core/kv_cache.*": [
        "qwen3-0.6b", "qwen25vl-3b", "whisper-tiny", "bark-small",
    ],
    "include/trtmc/runtime/recurrent_state.*": [
        "mamba-130m", "rwkv-7-0.1b", "nemotron-h-8b",
    ],
    "src/runtime/plugins/shared/diffusion_helpers.*": [
        "flux-schnell", "wan21-t2v-1.3b", "z-image-turbo",
    ],
    "src/runtime/plugins/shared/audio_helpers.*": [
        "whisper-tiny", "bark-small", "magpie-tts-357m", "personaplex-7b",
    ],
    "src/runtime/plugins/shared/plugin_helpers.*": "ALL",

    # ── Infrastructure: full suite ──
    "include/trtmc/runtime/trt_module.*": "ALL",
    "src/runtime/trt/core/trt_module.*": "ALL",
    "include/trtmc/pipeline.h": "ALL",
    "src/runtime/pipeline_factory.cpp": "ALL",
    "src/runtime/pipeline_registry.cpp": "ALL",
    "src/bundle/bundle_view.*": "ALL",
    "src/bundle/bundle_format.*": "ALL",
    "src/runtime/trt/core/trt_common.*": "ALL",
}
```

### CI pipeline design

```
PR push
  │
  ├─ Always: cmake build + ctest (unit tests, ~60s)
  │
  ├─ Diff analysis: git diff --name-only origin/master...HEAD
  │    → look up each changed file in COVERAGE_MAP
  │    → union all required E2E models
  │    → if any file maps to "ALL", run full E2E
  │
  ├─ If subset: run only the selected E2E models (~5-15 min)
  │    pytest tests/test_e2e.py -k "model1 or model2 or model3" ...
  │
  └─ If "ALL": run full E2E (~2-3 hours)

Nightly (scheduled):
  └─ Full E2E suite, all 50 models, --rebuild-engines
     + CCN gate
     + performance regression (Tier 5)
```

### What this buys you

| Scenario | Before (monolithic) | After (plugin) |
|---|---|---|
| Tune decoder KV cache layout | Full E2E (~3h) | 3 decoder models (~15 min) |
| Add new diffusion model | Full E2E (~3h) | 1 new model (~10 min) |
| Fix Whisper mel extraction | Full E2E (~3h) | whisper-tiny (~5 min) |
| Modify `TrtModule::forward()` | Full E2E (~3h) | Full E2E (~3h) — correct |
| Perf-tune flux denoiser loop | Full E2E (~3h) | flux-schnell only (~10 min) |

### Enforcement

The coverage map must be maintained as part of the plugin architecture:

1. **Adding a new plugin** → add its entry to `COVERAGE_MAP` with at least
   one E2E model. CI rejects plugins with no coverage entry.
2. **Adding a new E2E manifest** → add the model to the appropriate plugin's
   coverage list.
3. **Adding a new shared utility** → add it to the map with all dependent
   plugins' models listed. Default to `"ALL"` if unsure — can be narrowed later.
4. **Nightly full suite** catches any gaps in the coverage map. If nightly
   fails but PR CI passed, the coverage map has a hole — fix it.

## Depends On

- TASK-01 (Whisper + Bark migration to TrtModule)
- TASK-02 (Magpie + Speech + Omni migration to TrtModule)
- TASK-03 (Diffusion migration to TrtModule)
- TASK-04 (Legacy backend deletion)

All four must be complete before starting Phase 1. The plugin architecture
refactors the clean, uniform codebase that TASK-01–04 produce.
