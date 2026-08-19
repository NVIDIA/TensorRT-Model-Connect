---
title: Model Families
---

A model family is a model-owned Python build package. Its
`families/<family>/MODEL.toml` supplies discovery metadata, and its local
modules own config adaptation, checkpoint mapping, graph construction, and
bundle metadata.

## What a family model does

Every family exposes one required `model.py`. It:

- Match a Hugging Face `model_type` or diffusers pipeline class.
- Load and normalize weights.
- Emit the concrete model-owned runtime strategy implemented by the matching
  C++ model DSO.
- Build engine plan bytes.
- Add tokenizer, vision, diffusion, or audio metadata.
- Provide quantization exclusions or calibration data.
- Own the complete config → weights → engines → bundle sequence.

## Native TensorRT families

Native family packages live in
`python/tensorrt_model_connect/families/<family>/`. At this revision there are
80 package manifests. Use the repository validator for the live inventory:

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

### LocateAnything task contract

The current LocateAnything runtime supports the model's fixed 448×448,
single-image slow/autoregressive path. It preserves `<ref>`, `<box>`, and
`<0>` through `<1000>` tokens, including four-coordinate boxes and
two-coordinate points. Use the task helpers in
`tensorrt_model_connect.families.locateanything.task_contract` to construct the
official detection, single/multi grounding, text, GUI, and pointing prompts or
to parse structured outputs.

`--generation-mode auto`, `ar`, `autoregressive`, and `slow` select this path.
`fast` and `hybrid` require Parallel Box Decoding and fail explicitly in this
runtime; they are not silently treated as AR. The model-owned E2E contract
contains both a box case and a point case, while `refcoco_grounding` supplies
dataset-backed IoU accuracy validation.

`MODEL.toml` bounds lookup through IDs, aliases, prefixes,
`architecture_patterns`, and `diffusion_pipeline_classes`. After selecting a
candidate, the resolver imports exactly
`tensorrt_model_connect.families.<family>.model`, requires `matches(config)`
and `build(model_dir, output_path, **options)`, and calls `build()` directly.
There is no package scan, package-level proxy, compatibility fallback, or
manifest-configured Python entrypoint.

## Qualified optimized implementations

After family resolution, the selected `model.py` owns any native-versus-
optimized choice. Qwen keeps its exact Edge-LLM profile selection in its own
module; other families execute their native recipe directly.
A provider profile must match the exact model ID, immutable revision, active
target, and requested options and must retain its qualification state and
semantic-source binding. A successful claim packages its implementation DSO
and opaque artifact tree into the bundle. This is family policy rather than a
second central router.

The current example is the Qwen TensorRT Edge-LLM adapter with three qualified
Qwen3/A100 SM80/FP16 profiles. These profiles do not add native strategy keys
or claim support for arbitrary Qwen checkpoints and targets.

{/* Collaborative review anchor: batch 2. */}
