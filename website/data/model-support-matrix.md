<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<!-- Source data for the Supported Models website table. -->

| Hugging Face model ID (`hf_id`, CLI input) | TRTMC profile | Build precision | Quantization | Platform specialization runtime provider | GB300 |
| --- | --- | --- | --- | --- | --- |
| `albert/albert-base-v2` | `albert-base` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/all-MiniLM-L6-v2` | `all-minilm-l6-v2` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/all-mpnet-base-v2` | `all-mpnet-base-v2` | `FP16` | None | — | 🟢 Green |
| `suno/bark` | `bark-large` | `FP32` | None | — | 🟢 Green |
| `suno/bark-small` | `bark-small` | `FP16` | None | — | 🟢 Green |
| `facebook/bart-base` | `bart-base` | `FP16` | None | — | 🟢 Green |
| `google-bert/bert-base-uncased` | `bert-base-uncased` | `FP16` | None | — | 🟢 Green |
| `BAAI/bge-small-en-v1.5` | `bge-small-en-v1.5` | `FP16` | None | — | 🟢 Green |
| `bigscience/bloom-560m` | `bloom-560m` | `FP32` | None | — | 🟢 Green |
| `almanach/camembert-base` | `camembert-base` | `FP16` | None | — | 🟢 Green |
| `nvidia/canary-1b-v2` | `canary-1b-v2` | `FP16` | None | — | 🟢 Green |
| `amazon/chronos-bolt-tiny` | `chronos-bolt-tiny-official` | `FP32` | None | — | 🟢 Green |
| `Salesforce/codegen-350M-mono` | `codegen-350m` | `FP16` | None | — | 🟢 Green |
| `YituTech/conv-bert-base` | `convbert-base` | `FP16` | None | — | 🟢 Green |
| `microsoft/deberta-base` | `deberta-base` | `FP16` | None | — | 🟢 Green |
| `deepseek-ai/DeepSeek-OCR-2` | `deepseek-ocr` | `FP16`<br />FP32 layers: `6, 7, 8, 9, 10, 11, 12` | None | — | 🟢 Green |
| `deepseek-ai/DeepSeek-V2-Lite` | `deepseek-v2-lite` | `FP16` | None | — | 🟢 Green |
| `yujiepan/deepseek-v3-tiny-random` | `deepseek-v2-tiny` | `FP16` | None | — | 🟢 Green |
| `distilbert/distilbert-base-uncased` | `distilbert-base-uncased` | `FP16` | None | — | 🟢 Green |
| `distilbert/distilgpt2` | `distilgpt2` | `FP16` | None | — | 🟢 Green |
| `facebook/dpr-ctx_encoder-single-nq-base` | `dpr-ctx-encoder` | `FP16` | None | — | 🟢 Green |
| `google/electra-base-discriminator` | `electra-base-discriminator` | `FP16` | None | — | 🟢 Green |
| `tiiuae/falcon-rw-1b` | `falcon-rw-1b` | `FP16` | None | — | 🟢 Green |
| `tiiuae/Falcon3-1B-Base` | `falcon3-1b` | `FP16` | None | — | 🟢 Green |
| `black-forest-labs/FLUX.2-dev` | `flux-2-dev` | `FP16` | None | — | 🟢 Green |
| `black-forest-labs/FLUX.2-dev` | `flux-2-dev-fp8` | `FP16` | `fp8_scales=data/flux2-fp8-scales.json` | — | 🟢 Green |
| `black-forest-labs/FLUX.1-schnell` | `flux-schnell` | `FP16` | None | — | 🟢 Green |
| `google/fnet-base` | `fnet-base` | `FP16` | None | — | 🟢 Green |
| `google/gemma-2-2b-it` | `gemma-2-2b` | `FP16` | None | — | 🟢 Green |
| `THUDM/glm-4-9b-hf` | `glm-4-9b` | `FP16` | None | — | 🟢 Green |
| `EleutherAI/gpt-neo-125m` | `gpt-neo-125m` | `FP16` | None | — | 🟢 Green |
| `openai/gpt-oss-20b` | `gpt-oss-20b` | `FP16` | None | — | 🟢 Green |
| `openai-community/gpt2` | `gpt2-125m` | `FP32` | None | — | 🟢 Green |
| `ibm-granite/granite-3.1-2b-base` | `granite-3.1-2b` | `FP16` | None | — | 🟢 Green |
| `internlm/internlm2-math-plus-1_8b` | `internlm2-1.8b` | `FP16` | None | — | 🟢 Green |
| `IFM/K2-Horizon-7B`<br />Revision: `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10` | `k2-horizon-7b` | `BF16` | None | — | 🟢 Green |
| `OpenGVLab/InternVL3-2B-hf` | `internvl3-2b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `OpenGVLab/InternVL3-8B-hf` | `internvl3-8b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `bytedance-research/Lance` | `lance-3b-x2t-image` | `BF16` | None | — | 🟢 Green |
| `nvidia/LocateAnything-3B` | `locateanything-3b` | `FP16` | None | — | 🟢 Green |
| `nvidia/magpie_tts_multilingual_357m`<br />Revision: `34d7e40da85cabc97f92198889b65cea27bc7fd1` | `magpie-tts-357m` | `FP32` | None | — | 🟢 Green |
| `state-spaces/mamba-130m-hf` | `mamba-130m` | `FP32` | None | — | 🟢 Green |
| `Helsinki-NLP/opus-mt-en-ru` | `marian-en-ru` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Minitron-4B-Depth-Base` | `minitron-4b-depth` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Minitron-4B-Width-Base` | `minitron-4b-width` | `FP32` (manifest default) | None | — | 🟢 Green |
| `mistralai/Mistral-7B-Instruct-v0.1` | `mistral-7b` | `FP16` | None | — | 🟢 Green |
| `ggml-org/stories15M_MOE` | `mixtral-stories-15m` | `FP16`<br />FP32 layers: `4` | None | — | 🟢 Green |
| `answerdotai/ModernBERT-base` | `modernbert-base` | `FP32` | None | — | 🟢 Green |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | `nemotron-3.5-asr-streaming-0.6b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | `nemotron-embed-vl-1b-v2` | `FP16` | None | — | 🟢 Green |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `nemotron-h-nano-9b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `nvidia/Nemotron-4-Mini-Hindi-4B-Base` | `nemotron-hindi-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/Nemotron-Labs-Diffusion-8B` | `nemotron-labs-diffusion-8b` | `BF16` | None | — | 🟢 Green |
| `nvidia/Nemotron-Mini-4B-Instruct` | `nemotron-mini-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | `nemotron-nano-4b` | `FP16` | None | — | 🟢 Green |
| `nvidia/llama-nemotron-rerank-vl-1b-v2` | `nemotron-rerank-vl-1b-v2` | `FP16` | None | — | 🟢 Green |
| `nvidia/nemotron-speech-streaming-en-0.6b` | `nemotron-speech-streaming-en-0.6b` | `FP16` | None | — | 🟢 Green |
| `facebook/nllb-200-distilled-600M` | `nllb-200-distilled-600m` | `FP16` | None | — | 🟢 Green |
| `allenai/OLMo-1B-hf` | `olmo-1b` | `FP16` | None | — | 🟢 Green |
| `allenai/OLMo-2-0425-1B` | `olmo2-1b` | `FP16` | None | — | 🟢 Green |
| `facebook/opt-125m` | `opt-125m` | `FP16` | None | — | 🟢 Green |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `paraphrase-multilingual-minilm-l12-v2` | `FP16` | None | — | 🟢 Green |
| `ibm-granite/granite-timeseries-patchtsmixer` | `patchtsmixer-granite-official` | `FP16` | None | — | 🟢 Green |
| `ibm-research/patchtst-etth1-regression-distribution` | `patchtst-etth1-regression-distribution` | `FP16`<br />FP32 layers: `5` | None | — | 🟢 Green |
| `ibm-granite/granite-timeseries-patchtst` | `patchtst-granite-official` | `FP16` | None | — | 🟢 Green |
| `nvidia/personaplex-7b-v1` | `personaplex-7b` | `FP16`<br />FP32 layers: `0, 1` | None | — | 🟢 Green |
| `microsoft/Phi-tiny-MoE-instruct` | `phi-moe` | `FP16` | None | — | 🟢 Green |
| `microsoft/Phi-3-mini-4k-instruct` | `phi3-mini` | `FP16` | None | — | 🟢 Green |
| `microsoft/Phi-4-multimodal-instruct` | `phi4-multimodal` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | `pixart-sigma-1024` | `FP16`<br />FP32 layers: `0` | None | — | 🟢 Green |
| `EleutherAI/pythia-70m` | `pythia-70m` | `FP32` | None | — | 🟢 Green |
| `Qwen/Qwen-Image` | `qwen-image` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen-Image-2512` | `qwen-image-2512` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen-Image-Edit-2511` | `qwen-image-edit-2511` | `BF16` | None | — | 🟢 Green |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `qwen25vl-3b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-fp16` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional dispatch targets: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-fp8` | `BF16` | `format=fp8`<br />`scale_source=modelopt`<br />`calibration_samples=64` | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional dispatch targets: Coming soon | 🟢 Green |
| `Qwen/Qwen3-0.6B` | `qwen3-0.6b-topp` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional dispatch targets: Coming soon | 🟢 Green |
| `Qwen/Qwen3-4B-Instruct-2507` | `qwen3-4b-instruct-2507` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Linux x86_64, NVIDIA A100 80GB PCIe (SM80), FP16<br />Additional dispatch targets: Coming soon | 🟢 Green |
| `Qwen/Qwen3-30B-A3B` | `qwen3-moe-30b-a3b` | `FP16` | None | — | 🟢 Green |
| `amd-quark/tiny-random-qwen3_moe` | `qwen3-moe-tiny-random` | `FP16` | None | — | 🟢 Green |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `qwen3-omni-30b-a3b-instruct` | `BF16` | None | — | 🔴 Red |
| `Qwen/Qwen3-VL-2B-Instruct` | `qwen3-vl-2b` | `FP16`<br />FP32 layers: `0, 1, 2` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `Qwen/Qwen3.5-9B` | `qwen35-9b` | `FP16` | None | TensorRT Edge-LLM<br />Qualified TRTMC dispatch target: Coming soon | 🟢 Green |
| `Qwen/Qwen3.8-27B` | `qwen38-27b` | `FP16` | None | — | 🟡 Yellow |
| `nvidia/Riva-Translate-4B-Instruct-v1.1` | `riva-translate-4b` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/roberta-base` | `roberta-base` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/roberta-large` | `roberta-large` | `FP16` | None | — | 🟢 Green |
| `RWKV/rwkv-4-169m-pile` | `rwkv-169m` | `FP32` | None | — | 🟢 Green |
| `facebook/sam-vit-base` | `sam-vit-base` | `FP16` | None | — | 🟢 Green |
| `facebook/sam3` | `sam3` | `FP32` | None | — | 🟢 Green |
| `Efficient-Large-Model/SANA-WM_bidirectional` | `sana-wm-bidirectional` | `BF16` | None | — | 🟡 Yellow |
| `nvidia/segformer-b0-finetuned-ade-512-512` | `segformer-b0-ade` | `FP16` | None | — | 🟢 Green |
| `stabilityai/stablelm-2-1_6b` | `stablelm2-1.6b` | `FP16` | None | — | 🟢 Green |
| `bigcode/starcoder2-3b` | `starcoder2-3b` | `FP16` | None | — | 🟢 Green |
| `google-t5/t5-small` | `t5-small` | `FP16` | None | — | 🟢 Green |
| `google/timesfm-2.0-500m-pytorch` | `timesfm-2.0-500m-official` | `FP32` | None | — | 🟢 Green |
| `timm/mobilenetv3_large_100.ra_in1k` | `mobilenetv3-large-100-ra-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/efficientnet_b0.ra_in1k` | `efficientnet-b0-ra-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/densenet121.ra_in1k` | `densenet121-ra-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/mnasnet_100.rmsp_in1k` | `mnasnet-100-rmsp-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/inception_v3.tv_in1k` | `inception-v3-tv-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/repvgg_a2.rvgg_in1k` | `repvgg-a2-rvgg-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/resnest50d.in1k` | `resnest50d-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/resnet50.a1_in1k` | `resnet50-a1-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/vgg16.tv_in1k` | `vgg16-tv-in1k` | `FP16` | None | — | 🟢 Green |
| `timm/vit_base_patch16_224.augreg_in21k_ft_in1k` | `timm-vit-base-p16-224-augreg-in21k-ft-in1k` | `FP16` | None | — | 🟢 Green |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `tinyllama-1.1b` | `FP16` | None | — | 🟢 Green |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `wan21-t2v-1.3b` | `FP16`<br />FP32 layers: `24` | None | — | 🟢 Green |
| `Wan-AI/Wan2.2-TI2V-5B`<br />Revision: `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | `wan22-ti2v-5b` | `BF16` | None | — | 🟢 Green |
| `openai/whisper-large-v3-turbo` | `whisper-large-v3-turbo` | `FP32` | None | — | 🟢 Green |
| `openai/whisper-tiny` | `whisper-tiny-fp16` | `FP16`<br />FP32 layers: `0` | None | — | 🟢 Green |
| `facebook/xglm-564M` | `xglm-564m` | `FP16` | None | — | 🟢 Green |
| `FacebookAI/xlm-roberta-base` | `xlm-roberta-base` | `FP16` | None | — | 🟢 Green |
| `xlnet/xlnet-base-cased` | `xlnet-base` | `FP16` | None | — | 🟢 Green |
| `Tongyi-MAI/Z-Image-Turbo` | `z-image-turbo` | `FP16`<br />FP32 layers: `2, 3, 4, 7, 8` | None | — | 🔴 Red |

<!-- Collaborative review anchor: batch 2. -->
