---
title: Glossary
---

Use this page whenever a tutorial uses an unfamiliar deployment or inference term.

## Inference Concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| Model | A learned function that maps input numbers to output numbers. | The architecture and weights released by a model author. |
| Training | The process that creates or updates weights from data. | Out of scope. TensorRT-Model-Connect starts after training is done. |
| Inference | Running trained weights on a new request. | The C++ runtime loads a bundle and runs `generate`, `transcribe`, `solve`, `segment`, or another task method. |
| Checkpoint | Saved model files from training or release. | Usually a Hugging Face directory with `config.json`, weights, tokenizer, and processor files. |
| Tensor | A typed rectangular block of numbers. | Engine inputs and outputs are tensors such as token IDs, masks, logits, pixels, or audio features. |
| Shape | Tensor dimensions. | Examples: `[batch, sequence]` for token IDs or `[channels, height, width]` for image data. |
| Token | A numeric ID representing part of text. | Prompts are tokenized before they enter the engine. Output token IDs are decoded back to text. |
| Logits | Raw next-token scores before sampling. | A decoder engine returns logits, and the model-owned sampler chooses the next token. |
| Sampler | The rule for choosing a token from logits. | Greedy, top-k, top-p, temperature, and seed settings control it. |
| Prefill | The first decoder pass over the prompt. | It fills attention state for all prompt tokens. |
| Decode | The repeated one-token generation loop. | It reuses cached state and appends one token at a time. |
| KV cache | Reusable attention key/value tensors. | It avoids recomputing the full prompt for every generated token. |
| EOS | End-of-sequence token. | Generation can stop when EOS is produced or when `max_new_tokens` is reached. |

## Deployment Concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| CUDA | NVIDIA GPU programming/runtime stack. | Needed by the C++ runtime and TensorRT execution. |
| TensorRT | NVIDIA inference compiler/runtime. | Build-time code creates engine plans; runtime code deserializes and executes them. |
| Engine plan | Serialized TensorRT execution artifact. | Stored in bundle sections such as `engine_plan`, `vision_engine_plan`, or `denoiser_plan`. |
| `.bundle` bundle | TensorRT-Model-Connect deployable artifact. | A container with metadata plus either native config/plans/assets or an optimized-runtime descriptor and integrity-bound embedded implementation tree. |
| Hugging Face model ID | A repo name such as `Qwen/Qwen3-0.6B`. | `trtmc build` resolves it to a local model directory, downloading files if needed. |
| Precision | Numeric format used by engine weights/activations. | `fp16` is common for fast GPU smoke tests; `fp32` is larger and usually slower. |
| Quantization | Lower-precision representation such as FP8 or INT4. | Reduces footprint or latency when supported by the family and backend. |
| DSO (dynamic shared object) | A Linux shared library loaded while a process is running. | Native bundles use installed model/backend DSOs; optimized bundles carry their exact implementation DSO. |
| Backend DSO | A runtime-loaded shared library. | `libtrtmc_backend_trt.so` and `libtrtmc_backend_trt_rtx.so` isolate TensorRT ABI-sensitive calls. |
| ABI | Binary compatibility contract between compiled code and libraries. | TensorRT version mismatches can prevent an engine from loading. |
| Qualified profile | A support statement for one exact tested tuple. | It binds a model revision, implementation, target hardware/software, and public options; it is not a promise for nearby models or machines. |

## Project Building Blocks

| Term | Plain meaning | In this project |
| --- | --- | --- |
| Python builder | Build-time conversion tool. | `trtmc build` reads checkpoints, honors a family-owned native default route when declared, otherwise tries one exact-qualified optimized provider before the native fallback, and writes `.bundle` bundles. |
| C++ runtime | Request-time execution library and CLI. | `trtmc`, source-built `./build/trtmc`, and `trtmc::load()` load bundles and run task APIs. |
| Family plugin | Python adapter for a model family. | Examples: `qwen`, `llama`, `whisper`, `flux`, `pixart`. It handles config and weights. |
| Runtime strategy | Model-owned native C++ dispatch key in bundle metadata. | Examples: `qwen_decoder_kv_cache`, `whisper_speech_to_text`, `diffusion_flux`, `diffusion_pixart`. Optimized-runtime bundles use `optimized_runtime.json` instead. |
| Optimized-runtime descriptor | Exact delegated implementation contract in a bundle. | `optimized_runtime.json` binds the implementation/profile and embedded artifact tree; it bypasses native strategy, model-plugin, and backend-DSO selection. |
| Task strategy | E2E/user-contract category shared by models with the same result shape. | Examples: `text_generation_causal`, `speech_to_text`, `vision_language_generation`, `diffusion_media_generation`. It does not select a runtime DSO. |
| Pipeline | Task-oriented runtime implementation. | A concrete `IPipeline` handles generation, transcription, segmentation, solve, or another task. |
| Registry | Lookup table for native runtime plugins. | On the native path, `PipelineRegistry` maps `runtime_strategy` to an `IPipelinePlugin`. |
| E2E manifest | Canonical test description. | Files in `tests/e2e/models/` define model IDs, task type, expected runtime strategy, prompts, and tolerances. |
| Oracle | Reference behavior used by validation. | Usually Hugging Face, Diffusers, NeMo, or another official implementation. |
| Tolerance | Allowed numerical difference from the oracle. | Needed because optimized engines may not match reference floating-point values bit-for-bit. |

## What This Project Is Not

TensorRT-Model-Connect is not a training framework. It does not update model weights.

It is not a general model-serving cluster like vLLM, SGLang, TGI, or Triton Server. It provides artifact build tools, a native runtime, and task APIs that can be embedded into deployment systems.

It is not an automatic converter for every Hugging Face repo. A model needs a
compatible native family/runtime strategy or an exact qualified optimized
provider profile; otherwise it needs extension work.

It is not fully portable across every GPU, CUDA, and TensorRT version once an engine has been built. The bundle records compatibility metadata, and the runtime checks that metadata before execution.

{/* Collaborative review anchor. */}
