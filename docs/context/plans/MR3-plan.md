# MR-3 Plan: Delete God Structs

## Branch: `MR3-delete-god-structs`
## Base: `MR2-extract-plugins` (commit ceaf186)

## Goal

Delete `FastPathModelConfig` (239 LOC, 232+ fields) and `BundleSections`
(40+ pointer fields). Each plugin will parse its own config from raw JSON
and extract its own sections via `find_section()`.

---

## Complete dependency audit

### Files referencing FastPathModelConfig (26 files)

**Definition files (to be deleted):**
- `src/cabi/config/fast_path_config.h` — struct definition (239 LOC)
- `src/cabi/config/fast_path_config.cpp` — parse_fast_path_config() + 23 strategy parsers (419 LOC)

**Shared helper files (to be modified):**
- `src/runtime/plugins/shared/plugin_helpers.h` — includes header, compute_kv_dim(cfg), make_recurrent_gen_config(cfg)
- `src/runtime/plugins/shared/plugin_helpers.cpp` — implements above, all fields are in BaseConfig
- `src/runtime/plugins/shared/audio_helpers.h` — 6 functions take FastPathModelConfig
- `src/runtime/plugins/shared/audio_helpers.cpp` — implementations read 50+ strategy-specific fields
- `src/runtime/plugins/shared/diffusion_helpers.h` — make_diffusion_config(cfg), load_diffusion_parts(sections, cfg, ...)
- `src/runtime/plugins/shared/diffusion_helpers.cpp` — reads 30+ diffusion fields

**Plugin files (13 call parse_fast_path_config):**
- decoder_plugin.cpp, ssm_plugin.cpp, rwkv_plugin.cpp, hybrid_plugin.cpp
- vl_plugin.cpp, whisper_plugin.cpp, bark_plugin.cpp, magpie_plugin.cpp
- speech_plugin.cpp, omni_plugin.cpp, flux_plugin.cpp, wan_plugin.cpp, zimage_plugin.cpp

**Other (3 — not plugins that call it):**
- encoder_plugin.cpp, segmentation_plugin.cpp, object_detection_plugin.cpp — do NOT use it

**Infrastructure:**
- `src/runtime/pipeline_factory.cpp` — includes header but doesn't use it
- `src/cabi/bundle/bundle_helpers.h` — includes header for make_decoder_engine(cfg)

**Tests:**
- `tests/cpp/test_fast_path_config.cpp` — tests parse_fast_path_config()
- `tests/cpp/test_bundle_helpers.cpp` — creates FastPathModelConfig instances
- `tests/cpp/test_strategy_builder_helpers.h` — dead code, unreferenced

### Files referencing BundleSections (24 files)

**Definition files (to be deleted):**
- `src/cabi/bundle/bundle_helpers.h` — struct + find_bundle_sections + MelFilterbank + TokenizerResult + extract_tokenizer_from_bundle + extract_clip_tokenizer_from_bundle + load_mel_filterbank + make_decoder_engine
- `src/cabi/bundle/bundle_helpers.cpp` — implementations (328 LOC)

**Shared helper files:**
- `src/runtime/plugins/shared/plugin_helpers.h` — includes header, 5 functions take BundleSections
- `src/runtime/plugins/shared/plugin_helpers.cpp` — detect_add_special_tokens, is_bpe_tokenizer_json, try_create_native_bpe, create_tokenizer_from_bundle, load_trt_module
- `src/runtime/plugins/shared/audio_helpers.h` — 3 functions take BundleSections
- `src/runtime/plugins/shared/audio_helpers.cpp` — load_depth_engines, make_ipa_tok, build_speech_config_from_bundle
- `src/runtime/plugins/shared/diffusion_helpers.h` — load_diffusion_parts takes BundleSections
- `src/runtime/plugins/shared/diffusion_helpers.cpp` — reads denoiser/vae/text_encoder/preprocessor sections

**All 16 plugin files:** Each calls `find_bundle_sections(ctx.bundle)` + accesses specific fields.

**Audio pipeline (non-plugin):**
- `src/runtime/pipelines/audio_pipeline.h` — includes header for MelFilterbank (forward-declared, used as member)
- `src/runtime/pipelines/audio_pipeline.cpp` — includes header (for MelFilterbank)

**Audio validation:**
- `src/runtime/trt/audio/audio_bundle_validation.h` — validate function takes BundleSections
- `src/runtime/trt/audio/audio_bundle_validation.cpp` — reads specific BundleSections fields

**Tests:**
- `tests/cpp/test_bundle_helpers.cpp` — tests find_bundle_sections
- `tests/cpp/test_audio_bundle_validation.cpp` — creates BundleSections for tests

---

## Implementation strategy: Two-phase approach

### Phase A: Eliminate BundleSections (Commit 1)

**Rationale:** BundleSections is the easier struct to eliminate because each
plugin just needs to replace `sections.foo_data` with `find_section(ctx.bundle, "foo")`.

**Step A1: Move orphaned types out of bundle_helpers.h**

These types are defined in bundle_helpers.h but are NOT BundleSections-specific:
- `MelFilterbank` — move to `src/runtime/plugins/shared/plugin_helpers.h`
  (also move `load_mel_filterbank()` implementation)
- `TokenizerResult` — move to `src/runtime/plugins/shared/plugin_helpers.h`
  (also move `extract_tokenizer_from_bundle()` and `extract_clip_tokenizer_from_bundle()`)
- `make_decoder_engine()` — check if still used; if not, delete

After moving, `audio_pipeline.h` includes `plugin_helpers.h` instead of `bundle_helpers.h`.
Audio validation gets its own adapted signatures.

**Step A2: Adapt shared helper functions to take BundleFile&**

Change signatures in plugin_helpers.h:
```
BEFORE                                         AFTER
──────                                         ─────
load_trt_module(sections)                   →  (deleted — plugins use load_trt_module_from_plan directly)
detect_add_special_tokens(sections)         →  detect_add_special_tokens(bundle)
is_bpe_tokenizer_json(sections)             →  is_bpe_tokenizer_json(bundle)
try_create_native_bpe(sections, ...)        →  try_create_native_bpe(bundle, ...)
create_tokenizer_from_bundle(sections, ...) →  create_tokenizer_from_bundle(bundle, ...)
extract_tokenizer_from_bundle(sections,...) →  extract_tokenizer_from_bundle(bundle, ...)
extract_clip_tokenizer_from_bundle(s, ...)  →  extract_clip_tokenizer_from_bundle(bundle, ...)
```

Change signatures in audio_helpers.h:
```
load_depth_engines(sections, stream)        →  load_depth_engines(bundle, stream)
make_ipa_tok(sections)                      →  make_ipa_tok(bundle)
build_speech_config_from_bundle(sections, cfg, ...) →  build_speech_config_from_bundle(bundle, cfg, ...)
```

Change signatures in diffusion_helpers.h:
```
load_diffusion_parts(sections, cfg, ...)    →  load_diffusion_parts(bundle, cfg, ...)
```

Each implementation internally calls `find_section(bundle, "name")` instead of
`sections.name_data`.

**Step A3: Update all 16 plugins**

Each plugin changes from:
```cpp
auto sections = find_bundle_sections(ctx.bundle);
auto loaded = load_trt_module(sections);
auto tok = create_tokenizer_from_bundle(sections, ctx.hf_python);
```
To:
```cpp
auto* plan = find_section(ctx.bundle, "engine_plan");
auto loaded = load_trt_module_from_plan(plan, "engine_plan");
auto tok = create_tokenizer_from_bundle(ctx.bundle, ctx.hf_python);
```

Specific section access changes per plugin:
- `sections.plan_data` → `find_section(ctx.bundle, "engine_plan")`
- `sections.vision_plan_data` → `find_section(ctx.bundle, "vision_engine_plan")`
- `sections.coarse_engine_plan_data` → `find_section(ctx.bundle, "coarse_engine_plan")`
- `sections.fine_engine_plan_data` → `find_section(ctx.bundle, "fine_engine_plan")`
- `sections.codec_engine_plan_data` → `find_section(ctx.bundle, "codec_engine_plan")`
- `sections.denoiser_plan_data` → `find_section(ctx.bundle, "denoiser_plan")`
- `sections.vae_decoder_plan_data` → `find_section(ctx.bundle, "vae_decoder_plan")`
- `sections.text_encoder_plans` → `find_sections_by_prefix(ctx.bundle, "text_encoder_")`
- `sections.depth_engine_plans` → `find_sections_by_prefix(ctx.bundle, "depth_engine_plan_")`
- `sections.depth_engine_plan_data` → `find_section(ctx.bundle, "depth_engine_plan")`
- `sections.semantic_embed_data` → `find_section(ctx.bundle, "semantic_embed")`
- `sections.coarse_embed_data` → `find_section(ctx.bundle, "coarse_embed")`
- etc. (refer to kSectionMappings in bundle_helpers.cpp for the name↔field map)

**Step A4: Update audio_bundle_validation**

Change `validate_text_to_audio_bundle_sections()` to take `BundleFile&` and use
`find_section()` internally. Update test file accordingly.

**Step A5: Delete BundleSections**

Remove from `bundle_helpers.h`:
- `struct BundleSections` (lines 23-86)
- `BundleSections find_bundle_sections(const BundleFile& bundle)` (line 89)

Remove from `bundle_helpers.cpp`:
- `kSectionMappings[]` table (lines 23-76)
- `assign_mapped_section()` (lines 81-92)
- `assign_depth_engine_plan_section()` (lines 94-109)
- `is_text_encoder_plan_section()` (lines 111-115)
- `has_non_empty_data()` (lines 117-120)
- `has_bundle_tokenizer_data()` (lines 122-127)
- `create_tokenizer_temp_dir()` (lines 129-138)
- `write_optional_section_file()` (lines 140-157)
- `write_bundle_tokenizer_files()` (lines 159-170)
- `find_bundle_sections()` (lines 174-187)

If all functions are moved, delete both files entirely.

**Verification A:**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

---

### Phase B: Eliminate FastPathModelConfig (Commit 2)

**Rationale:** Now that BundleSections is gone, FastPathModelConfig is the
last god struct. Plugins need to parse strategy-specific fields from raw JSON.

**Key insight:** `compute_kv_dim()` and `make_recurrent_gen_config()` only
use BaseConfig fields. They can simply take `const BaseConfig&` instead.

**Step B1: Change plugin_helpers functions to use BaseConfig**

```
BEFORE                                    AFTER
──────                                    ─────
compute_kv_dim(FastPathModelConfig&)   →  compute_kv_dim(const BaseConfig&)
make_recurrent_gen_config(FPM&)        →  make_recurrent_gen_config(const BaseConfig&)
```

These are trivial — the fields they access (attention_size, head_dim, num_heads,
hidden_size, vocab_size, id_bos, id_eos) are ALL in BaseConfig already.

**Step B2: Change audio/diffusion helpers to parse from raw JSON**

For `make_diffusion_config(cfg)`: Change to `make_diffusion_config(const std::string& json)`.
Internally use `extract_json_string`, `extract_json_int`, `extract_json_float`,
`extract_json_float_array`, `extract_json_int_array` from `utils/json_helpers.h`.

For `build_magpie_config(cfg)`: Change to `build_magpie_config(const std::string& json, const BaseConfig& base)`.
Read magpie_* fields from JSON, use base.hidden_size as fallback.

For `build_speech_config_from_bundle(bundle, cfg, hf_python)`:
Change to `build_speech_config_from_bundle(const BundleFile& bundle, const std::string& json, const BaseConfig& base, const std::string& hf_python)`.

For `make_coarse_kv_cache(cfg, stream)`:
Change to `make_coarse_kv_cache(const std::string& json, const BaseConfig& base, cudaStream_t stream)`.

For `infer_speech_vocab_sizes(sc, cfg)`:
Change to `infer_speech_vocab_sizes(SpeechConfig& sc, const std::string& json, const BaseConfig& base)`.

For `make_depth_engine_config(cfg)`:
Change to return a small struct with just the fields needed (not FastPathModelConfig).
Or just inline it in the speech plugin.

For `compute_kv_dim_kv_heads(cfg, default_dim)`:
Change to `compute_kv_dim_kv_heads(const BaseConfig& base, int32_t default_dim)`.
The fields used (attention_size, num_kv_heads, head_dim) are ALL in BaseConfig.

**Step B3: Update all 13 plugins that call parse_fast_path_config()**

Each plugin changes from:
```cpp
auto cfg = parse_fast_path_config(ctx.config_json, ctx.config.max_cache_length);
```
To direct JSON extraction:
```cpp
const auto& json = ctx.config_json;
const auto& base = ctx.config;
int32_t d_inner = extract_json_int(json, "intermediate_size", 0);
if (d_inner == 0) d_inner = extract_json_int(json, "d_inner", base.hidden_size * 2);
// etc.
```

**Plugin complexity breakdown:**

| Plugin | Effort | Strategy-specific fields needed |
|--------|--------|-------------------------------|
| decoder | Trivial | None beyond BaseConfig |
| ssm | Easy | d_inner, state_size, conv_kernel (3 fields) |
| rwkv | Trivial | None beyond BaseConfig (hidden_size) |
| hybrid | Medium | num_mamba_layers, num_attention_layers, d_inner, mamba_d_state, mamba_d_conv, mamba_nheads, mamba_head_dim, conv_dim (8 fields) |
| vl | Medium | image_token_id, vision_output_dim, has_vision_engine, embed_input, fixed_image_size, num_image_pad_tokens, vl_prompt_template, image_token_str (8 fields) |
| whisper | Medium | num_mel_bins, max_source_positions, max_target_positions, encoder_layers, decoder_layers, eot_token_id, mel_length, mel_n_fft, mel_hop_length, mel_chunk_length, mel_sampling_rate, decoder_start_token_ids (12 fields) |
| bark | Hard | 25+ fields (sample_rate, semantic_*, coarse_*, fine_*, codec_*) |
| magpie | Medium | Delegates to build_magpie_config() which reads 15 fields |
| speech | Hard | Delegates to build_speech_config_from_bundle() which reads 20+ fields |
| omni | Medium | omni_* fields (12 fields) |
| flux/wan/zimage | Medium | Delegates to make_diffusion_config() which reads 30 fields |

**Step B4: Delete FastPathModelConfig files**

- Delete `src/cabi/config/fast_path_config.h`
- Delete `src/cabi/config/fast_path_config.cpp`
- Remove from CMakeLists.txt
- Remove `#include "cabi/config/fast_path_config.h"` from:
  - plugin_helpers.h
  - pipeline_factory.cpp

**Step B5: Update/delete tests**

- `test_fast_path_config.cpp` — DELETE (parse_fast_path_config no longer exists)
- `test_bundle_helpers.cpp` — DELETE or rewrite (BundleSections gone from Phase A)
- `test_strategy_builder_helpers.h` — DELETE (dead code)

**Verification B:**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
python tools/check_cyclomatic_complexity.py src --max-ccn 10
python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
python -m pytest tests/tools/ -v
```

---

## Risk matrix

| Step | Risk | Concern | Mitigation |
|------|------|---------|------------|
| A1 | Low | Moving MelFilterbank/TokenizerResult to wrong header | Check all includers |
| A2 | Medium | Adapting helpers to BundleFile& | Mechanical; each function just calls find_section() |
| A3 | Medium | Missing section name strings | Cross-reference kSectionMappings[] for exact names |
| A4 | Low | Audio validation is small | Only 2 validation functions |
| A5 | Low | Dangling includes | Grep for deleted header name |
| B1 | Low | BaseConfig has all needed fields | Already verified |
| B2 | Hard | Complex helpers (diffusion, speech) | 30+ extract_json_* calls each; tedious but safe |
| B3 | Hard | 13 plugins × variable complexity | Bark/speech plugins read 20+ fields from JSON |
| B4 | Low | Deleting files | Grep confirms no other consumers |
| B5 | Low | Test deletion | Coverage shifted to plugin tests + E2E |

## Estimated effort

- Phase A (BundleSections): ~2 hours (mechanical section name replacement)
- Phase B (FastPathModelConfig): ~3 hours (many extract_json_* calls, complex audio plugins)
- Testing + debugging: ~1 hour

Total: ~6 hours

## Recommended execution order

1. Phase A first (it's lower risk and gives a clean intermediate state)
2. Build + test at Phase A boundary
3. Phase B second (higher complexity, build on clean Phase A state)
4. Full verification at the end
