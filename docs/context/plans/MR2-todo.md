# MR-2 TODO: Extract All 16 Plugins

## Immediate next steps

1. **Create shared utility header** `src/runtime/plugins/shared/plugin_helpers.h`
   - Relocate from pipeline_factory.cpp anonymous namespace:
     - `LoadedModule` struct
     - `load_trt_module_from_plan()`
     - `try_load_trt_module_from_plan()`
     - `extract_optional_module()`
     - `create_tokenizer_from_bundle()`
     - `detect_add_special_tokens()`
     - `try_create_native_bpe()`
     - `is_bpe_tokenizer_json()`
     - `compute_kv_dim()`
     - `section_to_floats()`
     - `section_to_int32s()`
     - `has_section_data()`
     - `make_recurrent_gen_config()`

2. **Create decoder_plugin.cpp** as proof-of-concept
   - Register for "decoder_kv_cache" and "decoder_moe"
   - Move `create_decoder_pipeline()` logic
   - Add registry-first lookup to `from_bundle()`

3. **Extract remaining plugins** (Phases 4-7)
   - One plugin per strategy group
   - Each plugin is self-contained

4. **Shrink pipeline_factory.cpp** (Phase 8)
   - Remove all create_*() functions
   - Registry-only dispatch

## Key design decisions
- Use CMake OBJECT library for plugins to prevent linker stripping
- Shared utilities in `plugin_helpers.h` take `BundleFile&` and use `find_section()`
- Each plugin includes only the pipeline headers it needs
- `normalize_legacy_strategy()` stays in pipeline_factory.cpp until Phase 8

## Files to create (16 plugins + 3 shared headers)
```
src/runtime/plugins/
  shared/
    plugin_helpers.h        # TrtModule loading, tokenizer, kv_dim helpers
    diffusion_helpers.h     # DiffusionConfig, DiffusionParts, shared math
    audio_helpers.h         # Mel filterbank, cross-KV buffers, speech config
  decoder_plugin.cpp        # decoder_kv_cache, decoder_moe
  ssm_plugin.cpp            # ssm_recurrent
  rwkv_plugin.cpp           # rwkv_recurrent
  hybrid_plugin.cpp         # hybrid_mamba_attention
  encoder_plugin.cpp        # encoder_only, embedding, reranking, neural_operator
  segmentation_plugin.cpp   # segmentation, prompted_segmentation
  object_detection_plugin.cpp  # object_detection
  vl_plugin.cpp             # vision_language
  whisper_plugin.cpp        # speech_to_text
  bark_plugin.cpp           # text_to_audio_bark
  magpie_plugin.cpp         # text_to_audio_magpie
  speech_plugin.cpp         # speech_to_speech
  omni_plugin.cpp           # omni_multimodal
  flux_plugin.cpp           # diffusion_flux
  wan_plugin.cpp            # diffusion_wan
  zimage_plugin.cpp         # diffusion_zimage
  pixart_plugin.cpp         # diffusion_pixart (uses WanPipeline)
```

## Risks
- **Linker stripping**: Static initializers in plugins may be stripped if
  the TU has no external references. Mitigation: OBJECT library in CMake.
- **Include dependencies**: Plugins need both public (`include/trtmc/`) and
  private (`src/`) headers. Already handled by `trtmc_add_test()` pattern.
- **CCN**: Each plugin should stay under CCN=10. Complex plugins (bark, speech)
  may need internal helper functions.
