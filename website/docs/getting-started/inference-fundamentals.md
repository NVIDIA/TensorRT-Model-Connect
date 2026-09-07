---
title: Inference Fundamentals
---

import Diagram from '@site/src/components/Diagram';

This page explains the vocabulary behind TensorRT-Model-Connect. It assumes no
prior deep-learning inference background.

## Learning objectives

By the end of this module, you should be able to:

- distinguish a checkpoint, TensorRT engine, and `.bundle`;
- explain tokens, logits, prefill, decode, and KV cache;
- describe what belongs to shared core and what belongs to one family; and
- distinguish family, task, and backend identity.

## Training versus inference

Training creates model weights. Inference keeps those weights fixed and uses
them to answer a new request. Training includes forward computation, a loss,
and a backward pass that updates weights. Inference needs only forward
computation.

<Diagram
  src="/img/diagrams/getting-started/training-vs-inference.svg"
  alt="Training produces fixed checkpoint weights that inference uses to answer new requests"
  caption="TensorRT-Model-Connect begins at the trained checkpoint and owns the deployable inference path, not model training."
/>

## Checkpoints and engines

A Hugging Face-style checkpoint normally contains:

| File or concept | Meaning |
| --- | --- |
| `config.json` or `model_index.json` | Architecture or pipeline metadata used to identify the family. |
| Weight files | Learned tensors, often in safetensors or sharded files. |
| Tokenizer files | Rules for converting text to token IDs and back. |
| Processor files | Image, audio, or feature-extraction settings. |

A checkpoint is a portable model description plus weights. A TensorRT engine
is a compiled execution plan for a bounded GPU, CUDA, TensorRT, shape, and
precision target.

## Tensors and shapes

A tensor is a typed rectangular block of numbers. Common shapes include:

| Shape | Typical meaning |
| --- | --- |
| `[sequence]` | Token IDs for one prompt. |
| `[batch, sequence]` | Token IDs for several independent prompts. |
| `[batch, sequence, hidden]` | Embeddings or hidden states. |
| `[channels, height, width]` | Image pixels or image features. |
| `[frames, features]` | Audio features over time. |

At runtime, `trtmc::Tensor` is a non-owning tensor view,
`trtmc::DeviceTensor` owns GPU storage, and `TensorMap` associates engine tensor
names with values. Each family owns the exact names, shapes, bindings, and
pre/postprocessing used by its engines.

## Tokens, logits, and sampling

Language models operate on token IDs rather than strings. A tokenizer maps a
prompt to IDs. The model returns a logit for each possible next token, and a
sampler selects one token. Greedy sampling chooses the highest score; top-k,
top-p, temperature, and seed can produce controlled variation.

<Diagram
  src="/img/diagrams/trtmc-inference-loop.svg"
  alt="Text generation prefill and decode loop showing tokenization, KV cache, logits, sampling, and TextResult"
  caption="Prefill reads the prompt once; decode reuses the KV cache while the sampler selects each next token."
/>

## Prefill, decode, and KV cache

Autoregressive generation has two phases:

1. Prefill reads the complete prompt and creates attention state.
2. Decode generates one token at a time while reusing that state.

The reusable keys and values are the KV cache. Without it, every new token
would recompute attention over the full prompt and all previously generated
tokens. The owning family decides how its prefill/decode engines, cache,
sampling, and stopping behavior work.

## What TensorRT changes

General-purpose frameworks execute flexible operators. TensorRT compiles a
bounded graph into an engine plan optimized for GPU inference:

| Framework checkpoint | TensorRT engine |
| --- | --- |
| Portable when the matching Python stack is available. | Built for a particular compatibility and shape target. |
| Flexible and easy to inspect interactively. | Optimized and intentionally less dynamic. |
| Loads framework model code and weights. | Deserializes engine bytes and binds named tensors. |

TensorRT-Model-Connect keeps checkpoint-facing construction in Python and
request-time execution in native C++.

### Reference and deployment have different roles

| Concern | Framework reference | TensorRT-Model-Connect |
| --- | --- | --- |
| Model selection | Framework metadata chooses framework classes. | Dependency-free `support.py` declarations must identify exactly one family. |
| Build | Framework modules load original weights. | The selected family's plain `model.py` builds TensorRT sections. |
| Artifact | Checkpoint files and framework state. | A family-owned `.bundle`. |
| Runtime | Framework execution. | The loader opens the bundle's backend and family DSO from an explicit runtime root. |
| User behavior | Framework-specific calls. | The family implements an abstract Task interface. |
| Validation | Provides the expected result. | Model-owned tests compare the deployment result with meaningful thresholds. |

A successful build or plausible result is not parity evidence. The owning
family must select the reference, input, comparator, and threshold that prove
the claim.

## Why bundles exist

The bundle is the handoff between build and runtime. Its shared header contains
only:

- `format`;
- `family`;
- `task`; and
- `backend`.

The remainder is a bounded list of named byte sections. A family may write one
engine, separate prefill/decode engines, tokenizer assets, runtime metadata, or
other task-specific data. Shared core records section offsets and lengths but
does not interpret model-owned section names.

This keeps the container small in concept: build core calls the selected
family once, and runtime core loads the family named by the bundle once.

## Family, task, and backend

These identities answer different questions:

| Identity | Example | Question answered |
| --- | --- | --- |
| Family | `qwen`, `whisper`, `flux` | Which independent vertical slice owns build and runtime behavior? |
| Task | `text_generation`, `transcription`, `image_generation` | Which abstract user-facing interface must the family implement? |
| Backend | `trt`, `trt_rtx` | Which implementation of the Engine API executes serialized engines? |

The runtime loader does not contain a switch over model types. It opens the
exact family and backend named in the bundle. Different families can implement
the same Task API without depending on one another.

<Diagram
  src="/img/diagrams/getting-started/family-runtime-task-identity.svg"
  alt="Bundle family, task, and backend fields select separate model ownership, abstract user behavior, and engine implementation"
  caption="Family implementations depend on abstract Task and Engine contracts; those interfaces never depend on a concrete family."
/>

The family boundary includes `support.py`, a plain `model.py`, optional
`requirements.txt`, native runtime sources, and model-owned tests. The builder
function must not inherit from shared build machinery.

## Self-check

1. Why is a checkpoint not directly interchangeable with an engine or bundle?
2. Which identity chooses the model implementation, and which identity names
   user-visible behavior?
3. What does KV cache avoid recomputing?
4. Why can two text families implement the same Task API without sharing model
   code?

<details>
<summary>Check your answers</summary>

1. A checkpoint contains portable model metadata and weights; an engine is a
   compiled plan; a bundle packages family-owned runtime sections with a small
   routing header.
2. `family` selects the implementation and `task` names the abstract behavior.
3. It reuses attention keys and values for the prior prefix.
4. Both depend on the stable abstract interface, while their concrete build and
   runtime code remains in separate family directories.

</details>

## What to learn next

- Use [Quick Start](quick-start.md) to build and run one bundle.
- Use [Text Generation](../user-guides/text-generation.md) for request controls.
- Use the [Architecture](../architecture/ai-native-horizontal-scaling.md) to
  trace the exact dependency direction.
