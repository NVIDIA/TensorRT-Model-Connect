---
title: Runtime Strategies
---

This page describes native runtime strategies. A runtime strategy is the
model-owned C++ dispatch key stored in a native bundle's `config.json`. Each
key belongs to exactly one
`src/runtime/models/<owner>/MODEL.toml` and resolves to that owner's
`libtrtmc_model_<owner>.so`.

Optimized-runtime bundles are intentionally outside this inventory. They carry
`optimized_runtime.json` and an embedded implementation DSO, and the factory
selects that path before reading a native strategy.

## Strategy categories

| Category | Representative model-owned strategies |
| --- | --- |
| Text decoder | `qwen_decoder_kv_cache`, `llama_decoder_kv_cache`, `mixtral_decoder_moe`, `gpt_oss_decoder_moe` |
| Recurrent text | `mamba_ssm_recurrent`, `rwkv_recurrent`, `nemotron_h_hybrid_mamba_attention`, `qwen3_5_hybrid_mamba_attention`, `qwen3_8_hybrid_mamba_attention` |
| Encoder and retrieval | `bert_encoder_only`, `mpnet_encoder_only`, `eagle_vlm_embedding`, `eagle_vlm_reranking` |
| Seq2seq | `t5_text_to_text`, `marian_translation`, `bart_seq2seq_encoder_decoder`, `m2m_100_seq2seq_encoder_decoder` |
| Vision and multimodal | `qwen_vl_vision_language`, `internvl_vision_language`, `qwen3_omni_multimodal` |
| Speech and audio | `whisper_speech_to_text`, `nemotron_speech_streaming_speech_to_text_rnnt`, `text_to_audio_bark`, `personaplex_speech_to_speech` |
| Diffusion | `diffusion_flux`, `diffusion_wan`, `diffusion_wan2_2_ti2v`, `diffusion_qwen_image`, `diffusion_sana_wm` |
| Perception | `dinov3_image_feature_extraction`, `moge_monocular_geometry`, `segformer_segmentation`, `sam_prompted_segmentation`, `sam3_prompted_segmentation`, `timm_vit_image_classification`, `timm_resnet_image_classification`, `timm_mobilenetv3_image_classification`, `timm_vgg_image_classification` |
| Numeric operators | `chronos_bolt_trt`, `patchtsmixer_trt`, `patchtst_trt`, `timesfm_trt` |

The complete live list is the union of the `runtime_strategies` arrays in the
runtime model manifests; use `python3 tools/model_ci.py validate` rather than
copying snapshot counts into integrations.

## Runtime strategy versus task strategy

Do not use a generic task label as a runtime strategy:

| Layer | Qwen example | LLaMA example |
| --- | --- | --- |
| Python family | `qwen` | `llama` |
| Runtime strategy | `qwen_decoder_kv_cache` | `llama_decoder_kv_cache` |
| Runtime DSO | `libtrtmc_model_qwen.so` | `libtrtmc_model_llama.so` |
| E2E task strategy | `text_generation_causal` | `text_generation_causal` |

The task strategy lets generic runners and comparators share a user contract.
The runtime strategy keeps pipeline code, state, samplers, helpers, and
dependencies owned by the model.

{/* Collaborative review anchor: batch 2. */}
