---
title: Glossary
---

## Inference concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| Model | A learned function mapping inputs to outputs. | Architecture and trained weights released by a model author. |
| Training | Creating/updating weights from data. | Out of scope; Model Connect starts from a checkpoint. |
| Inference | Running fixed weights on a request. | A family Task implementation drives TensorRT engines. |
| Checkpoint | Saved model/config/tokenizer/processor files. | A Hugging Face ID or local snapshot consumed by one family builder. |
| Tensor | Typed multidimensional numbers. | Engine inputs/outputs such as token IDs, masks, logits, pixels, audio, or features. |
| Token and logits | Numeric text unit and raw next-token scores. | A family tokenizer maps text; a family sampler selects from logits. |
| Prefill and decode | Initial prompt pass and repeated next-token passes. | Family-owned text runtime orchestration. |
| KV cache | Reused attention keys/values. | Avoids recomputing all prior tokens on each decode step. |
| Oracle | Trusted expected behavior. | Usually an official framework/reference invoked by the family E2E test. |

## Deployment concepts

| Term | Plain meaning | In this project |
| --- | --- | --- |
| TensorRT engine plan | Compiled GPU execution artifact. | Opaque bytes stored in family-owned bundle sections. |
| `.bundle` | Build/runtime handoff container. | Format-1 header plus named sections; section semantics belong to the family. |
| Family | Unit of model ownership. | `families/<family>/` contains support, build, runtime, dependencies, and validation. |
| Task | User-visible behavior. | Abstract interfaces such as text generation, transcription, segmentation, embedding, and forecast. |
| Backend | Engine implementation. | `trt` or optional `trt_rtx`, selected by the bundle header. |
| DSO | Linux shared library loaded at runtime. | Core, loader, one backend, and exactly one `libtrtmc_model_<family>.so`. |
| Runtime root | Explicit DSO directory. | Required by every native execution command; no fallback search exists. |
| Precision | Numeric representation. | A build request such as FP32, FP16, or BF16 that the family validates. |
| Quantization | Lower-precision graph/weights such as FP8. | Entirely family-owned and qualified per exact checkpoint/path. |
| ABI | Binary compatibility contract. | Runtime DSOs and TensorRT plans must match their software/hardware cohort. |

## Project building blocks

| Term | Current role |
| --- | --- |
| `support.py` | Dependency-free exact checkpoint matching and supported/default tasks. |
| `model.py` | One plain family `build(request, writer)` implementation. |
| `BuildRequest` | Resolved typed build inputs passed to exactly one family. |
| `BundleWriter` / `BundleReader` | Streaming write and bounded read mechanics; no model semantics. |
| Family factory | `trtmc_create_family` in one family DSO, returning an abstract Task. |
| Engine API | Backend-neutral create/bind/enqueue contract used by family runtimes. |
| E2E manifest | Exact checkpoint, build, task, topology, selection, and testcase data owned by one family. |
| Threshold | Family-owned numeric override that must remain behaviorally meaningful. |

The former runtime-strategy registry, provider profiles, optimized-runtime
descriptor, shared config schemas, and task-strategy dispatch were removed by
PR #1093. Historical pages may mention them but current APIs do not.

TensorRT-Model-Connect is not a training framework, general serving cluster,
or automatic converter for arbitrary Hugging Face repositories. Unsupported
checkpoints need a complete family implementation and validation.
