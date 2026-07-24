---
title: Model Families
---

A model family is a model-owned Python build package. Its
`families/<family>/MODEL.toml` supplies discovery metadata, and its local
modules own config adaptation, checkpoint mapping, graph construction, and
bundle metadata.

## What a family plugin does

Family plugins can:

- Match a HuggingFace `model_type` or diffusers pipeline class.
- Load and normalize weights.
- Emit the concrete model-owned runtime strategy implemented by the matching
  C++ model DSO.
- Build engine plan bytes.
- Add tokenizer, vision, diffusion, or audio metadata.
- Provide quantization exclusions or calibration data.

## Native TensorRT families

Native family packages live in
`python/tensorrt_model_connect/families/<family>/`. At this revision there are
78 package manifests. Use the repository validator for the live inventory:

```bash
python3 tools/model_ci.py validate
```

Common native TensorRT groups:

- Decoder-only: Qwen, LLaMA, Mistral, GPT, OPT, Bloom, Gemma, Falcon, Granite, OLMo.
- MoE: Mixtral, Phi-MoE, Qwen-MoE, GPT-OSS, DeepSeek-V2.
- Recurrent and hybrid: Mamba, RWKV, Nemotron-H, Qwen3.5 hybrid.
- Encoder-only: BERT, RoBERTa, DeBERTa, ModernBERT, DistilBERT, ConvBERT, FNet, XLNet, MPNet, DPR.
- Seq2seq: T5, Marian, BART, M2M-100.
- Vision-language: Qwen-VL, InternVL, Lance, LocateAnything, Phi4 multimodal,
  DeepSeek-OCR.
- Audio and speech: Whisper, Canary, Bark, Magpie, PersonaPlex, Nemotron streaming.
- Diffusion: FLUX, Wan 2.1/2.2, LTX-Video, Qwen-Image, SANA-WM, Z-Image,
  PixArt.
- Perception: SegFormer, SAM, SAM3, and timm ViT classification.
- Time-series/operators: Chronos-Bolt, PatchTSMixer, PatchTST, and TimesFM.

The package's module-level `plugin` object still supplies the Python protocol,
but discovery begins with `MODEL.toml`; adding only a loose `.py` file does not
create a current family entry.
