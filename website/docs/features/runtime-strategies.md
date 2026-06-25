---
title: Runtime Strategies
---

A runtime strategy is the C++ dispatch key stored in bundle `config.json`.

## Strategy categories

| Category | Strategies |
| --- | --- |
| Text decoder | `decoder_kv_cache`, `decoder_moe` |
| Recurrent text | `mamba_ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` |
| Encoder and retrieval | `encoder_only`, `embedding`, `reranking`, `neural_operator` |
| Seq2seq | `t5_text_to_text`, `marian_translation`, `bart_seq2seq_encoder_decoder`, `m2m_100_seq2seq_encoder_decoder` |
| Vision and multimodal | `vision_language`, `omni_multimodal` |
| Speech and audio | `speech_to_text`, `speech_to_text_rnnt`, `text_to_audio_bark`, `text_to_audio_magpie`, `speech_to_speech` |
| Diffusion | `diffusion_flux`, `diffusion_wan`, `diffusion_zimage`, `diffusion_pixart` |
| Perception | `segmentation`, `prompted_segmentation`, `object_detection` |

## Why strategies are separate from families

Families describe how to build a model. Strategies describe how to run a bundle. This separation lets multiple families reuse one runtime path while still allowing specialized pipelines for genuinely different execution contracts.

Examples:

- Many decoder families reuse `decoder_kv_cache`.
- MoE decoder families reuse `decoder_moe`.
- SegFormer and SAM need different perception strategies.
- FLUX, Wan, Z-Image, and PixArt use separate diffusion strategies because their component layout and generation loops differ.
