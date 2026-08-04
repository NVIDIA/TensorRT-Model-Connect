---
title: Model Support Reference
---

import ModelSupportInventory from '@site/src/components/ModelSupportInventory';

This page is a support and evidence reference, not a prerequisite for your
first inference. New users should complete
[Getting Started](overview.md) before using this inventory to choose another
model.

Native model support is defined by three model-owned descriptor roots:

- Python build ownership:
  `python/tensorrt_model_connect/families/<family>/MODEL.toml`.
- C++ runtime ownership: `src/runtime/models/<runtime-owner>/MODEL.toml`.
- E2E contract ownership: `tests/e2e/models/<family>/MODEL.toml` and its
  declared `manifests/*.json`.

The published inventory summary is generated from those source trees at
documentation build time:

<ModelSupportInventory variant="facts" />

There is also an optimized-runtime path for exact qualified deployment tuples.
Its build ownership is under a selected family's adapter subtree; an
`IMPLEMENTATION.toml` identifies the delegated runtime and private
implementation DSO, while profile TOMLs bind an exact model ID, immutable
revision, target, build options, and qualification state. The current
implementation is Qwen with TensorRT Edge-LLM. Three Qwen3/A100 SM80/FP16
profiles retain exact qualification state and semantic-source bindings. Source
does not publish their former A100 hardware runner. They are not additional
`runtime_strategy` keys or blanket support for Qwen, x86_64, or A100.

Run the ownership validator instead of counting filenames manually:

```bash
python3 tools/model_ci.py validate
```

The command validates the descriptor alignment and prints the current logical
model inventory. It also handles the intentional physical-name differences
between `magpie_tts`/`magpie` and `wan_t2v`/`wan`.

## Family plugin inventory

The generated list below is a discovery inventory, not a claim that every
checkpoint in each family has current target-hardware proof:

<ModelSupportInventory variant="families" />

## Evidence levels

Keep these levels separate:

| Evidence | What it proves |
| --- | --- |
| Python family descriptor and plugin | The builder can recognize a family and has a model-owned implementation path. It does not prove a real checkpoint builds successfully. |
| Native runtime model descriptor and DSO source | The model owns a native runtime strategy, plugin registrar, and shared-library target. It does not prove a bundle executes on target hardware. |
| Optimized implementation manifest and profile | A family owns a delegated adapter and an exact model/revision/target/options tuple. Only a profile with current qualification state and semantic-source binding is eligible; the manifest alone is not qualification evidence. |
| E2E manifest | The repository declares a concrete model, task contract, input, oracle, and comparison policy. It is a test specification, not a current pass result. |
| Passing exact-revision E2E result | The named model contract passed for that revision and test environment. |
| Performance or promotion artifact | The measured model, bundle, hardware, software stack, command, and acceptance threshold passed. |

A Hugging Face repository that merely resembles a listed family is not
automatically supported. Its model type or pipeline class must resolve through
the family descriptor, and its exact checkpoint/configuration still needs
appropriate build and E2E evidence.

## Declared task contracts

The table groups current native E2E manifests by `task_strategy`. Native
runtime strategies remain model-owned; the examples are dispatch keys, not
generic aliases. Optimized-runtime profiles do not add rows to this native
strategy inventory. The current Source tree publishes no active
optimized-runtime producer.

| Task strategy | Example runtime strategies | Manifest families |
| --- | --- | --- |
| `text_generation_causal` | `qwen_decoder_kv_cache`, `mixtral_decoder_moe`, `mamba_ssm_recurrent`, `bart_seq2seq_encoder_decoder` | Decoder, MoE, recurrent/hybrid, seq2seq, and translation families. |
| `encoder_only_nlp` | `bert_encoder_only`, `roberta_encoder_only`, `xlnet_encoder_only` | ALBERT/BERT-style encoder families. |
| `embedding`, `reranking` | `eagle_vlm_embedding`, `eagle_vlm_reranking` | `eagle_vlm`. |
| `vision_language_generation` | `qwen_vl_vision_language`, `internvl_vision_language`, `locateanything_vision_language`, `deepseek_ocr_vision_language` | Qwen-VL, InternVL, Lance, LocateAnything, Phi4 multimodal, and DeepSeek-OCR. |
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

## How build and runtime dispatch are resolved

The Python builder reads family metadata from `families/*/MODEL.toml`, but its
three lookup flows are deliberately different:

1. A full config first narrows candidates with `architecture_patterns` and
   evaluates `matches_config()`. If no candidate resolves the config,
   `find_plugin()` uses the compatibility fallback: `pkgutil` imports every
   non-private family module/package and evaluates its predicates.
2. A string or `model_type` first tries a direct descriptor-ID lookup, then
   alias/prefix candidates, and finally the all-package `pkgutil` fallback.
3. A Diffusers pipeline class is matched only through descriptor
   `diffusion_pipeline_classes`; it imports matching packages and does not use
   the `pkgutil` fallback.

Each selected package exposes its package-level `plugin` from `__init__.py`.
The descriptor `module` field is specialization/tooling metadata, not an
arbitrary runtime-discovery selector. Loose `families/<family>.py` files can
therefore participate only in the two compatibility flows; they are not
complete support because the three ownership descriptors are still required.

After family resolution, a matching model-owned `default_build_route` may send
the request directly to the native `FamilyPlugin`; eligible dense Qwen3 and
Llama currently do so. Other requests probe optimized implementations only
inside that family's directory. Exactly one qualified profile may claim the
model revision, active target, and requested options. A successful claim writes
an optimized bundle with `optimized_runtime.json`, implementation metadata,
and an embedded `libtrtmc_impl_*.so`; no claim continues to the native build.

### Dense Qwen3 and Llama native default

For an architecture-compatible dense Qwen3 or Llama checkpoint, the family
route skips optimized providers. With no build overrides, the current native-KV
contract is:

| Property | Default contract |
| --- | --- |
| Precision | BF16 |
| KV capacity | Fixed at the checkpoint's complete `max_position_embeddings` |
| Engine layout | Split prefill and decode sections: `prefill_engine_plan` and `engine_plan` |
| Backend and topology | Standard TensorRT on one GPU, without tensor parallelism |
| Excluded options | Quantization, dynamic KV, FP32 layer overrides, and debug layer outputs |
| Bundle marker | `native_kv_cache=true` with native-KV contract version 1 |

An explicit request outside that deployment contract uses the legacy native
builder when the family supports it; it does not re-enter optimized-provider
selection. A native-KV bundle owns one fixed physical cache and rejects the
runtime `--kv-cache-size` override. Loading also performs a GPU-memory admission
check for the full cache, so a valid build can still require a GPU with enough
free memory at runtime.

For a native build, the bundle carries the plugin's concrete
`runtime_strategy`. At CMake configure time,
`cmake/trtmc_pipeline_plugins.cmake` discovers every
`src/runtime/models/*/MODEL.toml`, rejects duplicate strategy ownership, and
generates a strategy-to-model-DSO index. At load time, a native bundle uses
that index to load only the owning `libtrtmc_model_<owner>.so`, whose registrar
adds the requested `IPipelinePlugin` to `PipelineRegistry`.

For an optimized bundle, `PipelineFactory` sees `optimized_runtime.json`
before native config/strategy dispatch. It integrity-checks and materializes
the embedded artifact tree, loads the exact implementation DSO, validates its
private factory ABI and identities, and asks it to create the public
`IPipeline`. It does not consult the native strategy index or load a generic
backend DSO. An optimized-path error is terminal rather than a request to fall
back to the native bundle path.

For the exact checkpoint list, inputs, precision, task strategy, and oracle,
read the JSON manifests declared by the relevant
`tests/e2e/models/<family>/MODEL.toml`. For an optimized implementation,
inspect its family-owned profile TOML. The retained qualified profiles are
exact product routes; because Source does not publish their former
target-hardware runner, a Source checkout alone cannot reproduce their
hardware qualification.

{/* Collaborative review anchor. */}
