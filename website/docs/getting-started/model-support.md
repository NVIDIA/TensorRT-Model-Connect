---
title: Model Support
---

Model support is defined by two source-of-truth surfaces:

- Python family plugins under `python/tensorrt_model_connect/families/`.
- E2E manifests under `tests/e2e/models/`.

The current checkout contains 68 Python family plugins and 122 E2E model manifests.

## Support Levels

Use these terms when reading the tables:

| Level | Meaning |
| --- | --- |
| First-class E2E support | A manifest exists, the builder can create a bundle, the C++ runtime has a matching strategy, and the E2E harness validates behavior against a reference. |
| Builder support | A Python family plugin can create bundle artifacts, but runtime or E2E coverage may be narrower. |
| Runtime strategy support | A C++ plugin and pipeline shape exist for bundles that carry the strategy. Specific families still need builder and manifest coverage. |
| Experimental support | The path exists but may have stricter environment, model-size, precision, or parity limits. Read the manifest and tutorial caveats before relying on it. |
| Not supported by listing alone | A HuggingFace model name that resembles a family is not automatically supported. The family plugin and runtime strategy must match the model's config and request-time behavior. |

The E2E manifests are the most concrete proof because they name the model ID, runtime strategy, prompt/input shape, verifier, and tolerances.

## Supported task families

| Task | Runtime strategies | Example manifest families |
| --- | --- | --- |
| Decoder text generation | `decoder_kv_cache`, `decoder_moe` | `qwen`, `llama`, `mistral`, `gpt_oss`, `deepseek_v2`, `mixtral` |
| Recurrent and hybrid text | `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention` | `mamba`, `rwkv`, `nemotron_h`, `qwen3_5` |
| Encoder-only and retrieval | `encoder_only`, `embedding`, `reranking` | `bert`, `roberta`, `mpnet`, `eagle_vlm` |
| Seq2seq and translation | `text_to_text`, `marian_translation`, `seq2seq_encoder_decoder` | `t5`, `marian`, `bart`, `m2m_100` |
| Vision-language and OCR | `vision_language`, `omni_multimodal` | `qwen_vl`, `internvl`, `phi4_multimodal`, `deepseek_ocr`, `qwen3_omni` |
| Speech and audio | `speech_to_text`, `speech_to_text_rnnt`, `text_to_audio_bark`, `text_to_audio_magpie`, `speech_to_speech` | `whisper`, `canary`, `nemotron_speech_streaming`, `bark`, `magpie_tts`, `personaplex` |
| Diffusion image/video | `diffusion_flux`, `diffusion_wan`, `diffusion_zimage`, `diffusion_pixart` | `flux`, `wan_t2v`, `z_image`, `pixart` |
| Segmentation and detection | `segmentation`, `prompted_segmentation`, `object_detection` | `segformer`, `sam` |
| Operators | `neural_operator` | family-specific numeric operator bundles |

## Family plugin inventory

The current Python plugin inventory is:

```text
albert, bark, bart, bert, bloom, canary, codegen, convbert, deberta,
deepseek_ocr, deepseek_v2, distilbert, dpr, eagle_vlm, electra, falcon,
flux, fnet, gemma, glm, gpt2, gpt_neo, gpt_neox, gpt_oss, granite,
internlm, internvl, llama, m2m_100, magpie_tts, mamba, marian, mistral,
mixtral, modernbert, mpnet, nemotron, nemotron_h, nemotron_speech_streaming,
olmo, olmo2, opt, personaplex, phi, phi4_multimodal, phi_moe, pixart,
qwen, qwen3_5, qwen3_omni, qwen_moe, qwen_vl, roberta, rwkv, sam,
segformer, stablelm, starcoder2, t5, wan_t2v, whisper, xglm, xlnet,
z_image
```

## How support is resolved

For native TRT builds, `python/tensorrt_model_connect/families/__init__.py` scans every non-private family module and registers its module-level `plugin` object. Standard models use `find_plugin(model_type)`. Diffusion models use `find_diffusion_plugin(pipeline_class)`.

For runtime, the bundle's `runtime_strategy` selects a C++ `IPipelinePlugin` registered through `cmake/trtmc_pipeline_plugins.cmake`.

:::note CLI coverage is smaller than runtime coverage
Some runtime strategies have C++ API and test coverage before they have a polished CLI wrapper. Object detection is available through `trtmc detect`; less common strategies may still require the C++ API or dedicated tests until their CLI path is polished.
:::
