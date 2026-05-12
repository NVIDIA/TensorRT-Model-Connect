---
title: Model Families
---

A model family is the Python build-time owner for a group of HuggingFace models.

## What a family plugin does

Family plugins can:

- Match a HuggingFace `model_type` or diffusers pipeline class.
- Load and normalize weights.
- Choose a runtime strategy.
- Build engine plan bytes.
- Add tokenizer, vision, diffusion, audio, or time-series metadata.
- Provide quantization exclusions or calibration data.

## Raw TRT families

Raw TRT family plugins live in `tensorrt_model_connect/tensorrt_model_connect/families/`. The current checkout has 65 plugins.

Common raw TRT groups:

- Decoder-only: Qwen, LLaMA, Mistral, GPT, OPT, Bloom, Gemma, Falcon, Granite, OLMo.
- MoE: Mixtral, Phi-MoE, Qwen-MoE, GPT-OSS, DeepSeek-V2.
- Recurrent and hybrid: Mamba, RWKV, Nemotron-H, Qwen3.5 hybrid.
- Encoder-only: BERT, RoBERTa, DeBERTa, ModernBERT, DistilBERT, ConvBERT, FNet, XLNet, MPNet, DPR.
- Seq2seq: T5, Marian, BART, M2M-100.
- Vision-language: Qwen-VL, InternVL, Phi4 multimodal, DeepSeek-OCR.
- Audio and speech: Whisper, Canary, Bark, Magpie, PersonaPlex, Nemotron streaming.
- Diffusion: FLUX, Wan, Z-Image, PixArt.
- Perception: SegFormer, SAM.

## Torch-TRT families

Torch-TRT engine definition families live in `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/`. They are useful when a model can be captured through `torch.export` and compiled without hand-written graph construction.
