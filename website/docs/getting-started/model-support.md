---
title: Model Support
---

Model support is defined by three model-owned descriptor roots:

- Python build ownership:
  `python/tensorrt_model_connect/families/<family>/MODEL.toml`.
- C++ runtime ownership: `src/runtime/models/<runtime-owner>/MODEL.toml`.
- E2E contract ownership: `tests/e2e/models/<family>/MODEL.toml` and its
  declared `manifests/*.json`.

At this revision, those roots describe 78 logical model families, 78 runtime
model DSOs, 79 unique runtime strategy keys, 78 E2E family indexes, and 203
declared E2E manifests. Eagle VLM is the one runtime owner with two strategy
keys (`eagle_vlm_embedding` and `eagle_vlm_reranking`).

Run the ownership validator instead of counting filenames manually:

```bash
python3 tools/model_ci.py validate
```

The command validates the descriptor alignment and prints the current logical
model inventory. It also handles the intentional physical-name differences
between `magpie_tts`/`magpie` and `wan_t2v`/`wan`.

## Evidence levels

Keep these levels separate:

| Evidence | What it proves |
| --- | --- |
| Python family descriptor and plugin | The builder can recognize a family and has a model-owned implementation path. It does not prove a real checkpoint builds successfully. |
| Runtime model descriptor and DSO source | The model owns a runtime strategy, plugin registrar, and shared-library target. It does not prove a bundle executes on target hardware. |
| E2E manifest | The repository declares a concrete model, task contract, input, oracle, and comparison policy. It is a test specification, not a current pass result. |
| Passing exact-revision E2E result | The named model contract passed for that revision and test environment. |
| Performance or promotion artifact | The measured model, bundle, hardware, software stack, command, and acceptance threshold passed. |

A HuggingFace repository that merely resembles a listed family is not
automatically supported. Its model type or pipeline class must resolve through
the family descriptor, and its exact checkpoint/configuration still needs
appropriate build and E2E evidence.

## Declared task contracts

The table groups current E2E manifests by `task_strategy`. Runtime strategies
remain model-owned; the examples are dispatch keys, not generic aliases.

| Task strategy | Example runtime strategies | Manifest families |
| --- | --- | --- |
| `text_generation_causal` | `qwen_decoder_kv_cache`, `mixtral_decoder_moe`, `mamba_ssm_recurrent`, `bart_seq2seq_encoder_decoder` | Decoder, MoE, recurrent/hybrid, seq2seq, and translation families. |
| `encoder_only_nlp` | `bert_encoder_only`, `roberta_encoder_only`, `xlnet_encoder_only` | ALBERT/BERT-style encoder families. |
| `embedding`, `reranking` | `eagle_vlm_embedding`, `eagle_vlm_reranking` | `eagle_vlm`. |
| `vision_language_generation` | `qwen_vl_vision_language`, `internvl_vision_language`, `deepseek_ocr_vision_language` | Qwen-VL, InternVL, Lance, LocateAnything, Phi4 multimodal, and DeepSeek-OCR. |
| `omni_multimodal` | `qwen3_omni_multimodal` | `qwen3_omni`. |
| `speech_to_text` | `whisper_speech_to_text`, `canary_speech_to_text`, `nemotron_speech_streaming_speech_to_text_rnnt` | Whisper, Canary, and Nemotron streaming. |
| `text_to_audio`, `speech_to_speech` | `text_to_audio_bark`, `text_to_audio_magpie`, `personaplex_speech_to_speech` | Bark, Magpie TTS, and PersonaPlex. |
| `diffusion_media_generation` | `diffusion_flux`, `diffusion_wan`, `diffusion_wan2_2_ti2v`, `diffusion_qwen_image`, `diffusion_sana_wm` | Image and video diffusion families. |
| `diffusion_text_generation` | `elf_flow` | `elf_flow`. |
| `segmentation`, `prompted_segmentation` | `segformer_segmentation`, `sam_prompted_segmentation`, `sam3_prompted_segmentation` | SegFormer, SAM, and SAM3. |
| `image_classification` | `timm_vit_image_classification` | `timm_vit`. |
| `neural_operator` | `chronos_bolt_trt`, `patchtsmixer_trt`, `patchtst_trt`, `timesfm_trt` | Time-series/operator families. |

The public C++ API and CLI also expose a detection surface, but this revision
does not have a model-owned `object_detection` runtime strategy or E2E family.
An API method or CLI command alone is not model-support evidence.

## How dispatch is resolved

The Python builder reads family metadata from
`families/*/MODEL.toml`, narrows candidate packages by aliases, prefixes,
architecture patterns, or diffusion pipeline classes, then imports the
selected package's module-level `plugin`.

The built bundle carries that plugin's concrete `runtime_strategy`.
At CMake configure time, `cmake/trtmc_pipeline_plugins.cmake` discovers every
`src/runtime/models/*/MODEL.toml`, rejects duplicate strategy ownership, and
generates a strategy-to-model-DSO index. At load time, the runtime uses that
index to load only the owning `libtrtmc_model_<owner>.so`, whose registrar adds
the requested `IPipelinePlugin` to `PipelineRegistry`.

For the exact checkpoint list, inputs, precision, task strategy, and oracle,
read the JSON manifests declared by the relevant
`tests/e2e/models/<family>/MODEL.toml`.
