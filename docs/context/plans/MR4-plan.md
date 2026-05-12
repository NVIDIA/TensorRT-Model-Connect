# MR-4 Plan: Documentation Update

## Branch: TBD (can be done in parallel with MR-3)
## Base: `MR2-extract-plugins` or `MR3-delete-god-structs`

## Goal

Update documentation to reflect the new plugin architecture.

---

## Changes needed

### 1. CLAUDE.md — Source layout section

Add `src/runtime/plugins/` directory to the source layout tree:
```
src/runtime/
  plugins/                            # Self-registering pipeline plugins
    shared/
      plugin_helpers.h/cpp            # TrtModule loading, tokenizer, KV helpers
      diffusion_helpers.h/cpp         # Shared diffusion config/loading
      audio_helpers.h/cpp             # Audio config, mel, speech helpers
    decoder_plugin.cpp                # decoder_kv_cache, decoder_moe
    ssm_plugin.cpp                    # ssm_recurrent
    rwkv_plugin.cpp                   # rwkv_recurrent
    hybrid_plugin.cpp                 # hybrid_mamba_attention
    encoder_plugin.cpp                # encoder_only, embedding, reranking, neural_operator
    segmentation_plugin.cpp           # segmentation, prompted_segmentation
    object_detection_plugin.cpp       # object_detection
    vl_plugin.cpp                     # vision_language
    whisper_plugin.cpp                # speech_to_text
    bark_plugin.cpp                   # text_to_audio_bark
    magpie_plugin.cpp                 # text_to_audio_magpie
    speech_plugin.cpp                 # speech_to_speech
    omni_plugin.cpp                   # omni_multimodal
    flux_plugin.cpp                   # diffusion_flux
    wan_plugin.cpp                    # diffusion_wan, diffusion_pixart
    zimage_plugin.cpp                 # diffusion_zimage
    force_link_plugins.cpp            # Linker anchors for static lib
```

### 2. CLAUDE.md — "Adding a new model family" section

Update to explain the plugin-based workflow:
- Creating a new plugin .cpp file
- Using REGISTER_PIPELINE_PLUGIN macro
- Adding force-link anchor
- No edits to pipeline_factory.cpp needed

### 3. CLAUDE.md — Architecture section

Update to reflect:
- pipeline_factory.cpp is now 126 LOC (registry-based dispatch)
- Each strategy is a self-contained plugin
- BaseConfig replaces FastPathModelConfig (if MR-3 done)

### 4. Optional: ci/test_coverage_map.py

Create the coverage map from TASK-05 spec (maps changed files to minimum
E2E test set for CI optimization).

---

## Risk: None (docs only)

## Estimated effort: ~30 minutes
