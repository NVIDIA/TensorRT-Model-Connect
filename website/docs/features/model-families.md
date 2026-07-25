---
title: Model Families
---

A model family is a model-owned Python build package. Its
`families/<family>/MODEL.toml` supplies discovery metadata, and its local
modules own config adaptation, checkpoint mapping, graph construction, and
bundle metadata.

## What a family plugin does

On the native path, family plugins can:

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

The package-level `plugin` exported by `__init__.py` supplies the Python
protocol, while `MODEL.toml` indexes discovery. The lookup route depends on
the input: a full config tries bounded `architecture_patterns` candidates
before the all-package `pkgutil` compatibility fallback; a string or
`model_type` tries a direct descriptor ID, then alias/prefix candidates, then
that fallback; a Diffusers pipeline class uses descriptor
`diffusion_pipeline_classes` only and never runs the fallback. The descriptor
`module` field is specialization/tooling metadata, not a runtime import
selector. Adding only a loose `.py` file can therefore be seen by the two
compatibility flows, but it does not create a complete current family entry.

## Qualified optimized implementations

After family resolution, the public build path probes optimized
implementations only below that family. A provider profile must match the
exact model ID, immutable revision, active target, and requested options and
must retain its qualification state and semantic-source binding. A successful
claim packages its implementation DSO and opaque artifact tree into the
bundle; no claim continues to the native plugin above.

The current example is the Qwen TensorRT Edge-LLM adapter with three qualified
Qwen3/A100 SM80/FP16 profiles. These profiles do not add native strategy keys
or claim support for arbitrary Qwen checkpoints and targets.
