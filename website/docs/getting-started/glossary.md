---
title: Glossary
---

Use this page whenever a tutorial introduces an unfamiliar deployment or
inference term.

## Inference concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| Model | A learned function that maps input numbers to output numbers. | Architecture and weights released by a model author. |
| Training | Creating or updating weights from data. | Out of scope; TRTMC starts with a trained checkpoint. |
| Inference | Applying fixed weights to a new request. | A family implementation runs an abstract Task method. |
| Checkpoint | Saved model files. | Usually config, weights, tokenizer, and processor files in a Hugging Face snapshot. |
| Tensor | A typed rectangular block of numbers. | Named engine inputs and outputs such as token IDs, pixels, features, or logits. |
| Token | A numeric ID representing part of text. | A family tokenizer maps prompts to IDs and generated IDs back to text. |
| Logits | Scores for possible next tokens. | A text family uses them to choose the next token. |
| Prefill | The first decoder pass over a prompt. | It creates attention state for the prompt tokens. |
| Decode | Repeated one-token generation. | It reuses state and appends a token each step. |
| KV cache | Reusable attention keys and values. | It avoids recomputing the complete prefix on every decode step. |

## Deployment concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| TensorRT | NVIDIA inference compiler and runtime. | Family build code creates engines; a backend executes them. |
| Engine plan | Serialized TensorRT execution artifact. | Stored in one or more family-owned bundle sections. |
| `.bundle` | Build-to-runtime artifact. | A small header plus named byte sections written by one family. |
| Precision | Numeric format for weights or activations. | `fp16`, `bf16`, or `fp32`, when supported by the family and target. |
| Quantization | Lower-precision representation. | A build option whose implementation and support are family-owned. |
| DSO | A dynamic shared object loaded by a process. | The runtime root contains core/runtime, backend, and selected family libraries. |
| Backend | Implementation of the abstract Engine API. | `trt` and optional `trt_rtx` are selected by the bundle header. |
| ABI | Binary compatibility contract. | TensorRT or compiled-library mismatches can prevent loading. |

## Repository building blocks

| Term | Plain meaning | In this project |
| --- | --- | --- |
| `support.py` | Lightweight family declaration. | Claims checkpoint metadata and lists supported/default tasks without importing heavy family dependencies. |
| `model.py` | Concrete family builder. | A plain `build(request, writer)` function; builder inheritance is forbidden. |
| Family | One model-owned vertical slice. | Build code, runtime DSO, dependencies, bundle semantics, and tests under `families/<family>/`. |
| Bundle header | Shared routing information. | `format`, `family`, `task`, and `backend`; section meaning belongs to the family. |
| Task API | Abstract user-facing behavior. | Interfaces for text, embedding, transcription, media, segmentation, forecasting, and other tasks. |
| Engine API | Abstract engine execution behavior. | The family uses it without linking directly to a concrete backend implementation. |
| Runtime root | Explicit native-library directory. | Required by every execution command; the loader does not search elsewhere. |
| E2E manifest | Model-owned test description. | A file under `families/<family>/tests/manifests/` with a checkpoint, task, inputs, and build settings. |
| Oracle | Reference behavior used by validation. | Usually the official framework implementation. |
| Tolerance | Allowed numerical difference. | Chosen by the owning family for a meaningful comparator. |

## What this project is not

TRTMC is not a training framework, a general serving cluster, or an automatic
converter for every checkpoint. A checkpoint is supported only when exactly
one family claims it and owns a working build, runtime implementation, and
validation contract.

Engine portability is bounded by the GPU, CUDA, TensorRT, and compiled-library
cohort used by the selected family.
