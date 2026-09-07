---
title: Inference Fundamentals
---

This page explains the vocabulary behind TensorRT-Model-Connect. It assumes no
prior deep-learning inference background.

## Training, checkpoints, and inference

Training creates model weights. Inference keeps those weights fixed and runs a
forward computation for a new request. TensorRT-Model-Connect starts at a
trained checkpoint and owns the build-to-native-inference path, not training.

A Hugging Face-style checkpoint usually contains architecture configuration,
weight tensors, tokenizer files, and modality preprocessors. It is not a
TensorRT engine: the checkpoint is portable source material, while an engine
is a compiled plan for a particular TensorRT/CUDA/GPU cohort.

## Tensors, tokens, and shapes

A tensor is a typed multidimensional array. Common shapes include token IDs
`[batch, sequence]`, hidden states `[batch, sequence, hidden]`, images
`[channels, height, width]`, and audio features `[frames, features]`.

Text models convert strings into token IDs. A decoder produces one score per
possible next token; these scores are logits. The family-owned sampler applies
greedy or stochastic settings and chooses the next token.

## Prefill, decode, and KV cache

Autoregressive generation has two phases:

1. prefill reads the complete prompt and creates attention state;
2. decode generates one token at a time while reusing that state.

The reusable key/value attention tensors are the KV cache. Without it, every
decode step would recompute the entire prefix. Each family owns its tokenizer,
state, cache layout, sampler, stopping logic, and request loop.

## What TensorRT changes

TensorRT compiles a fixed network into a GPU execution plan. Compared with an
eager framework checkpoint, the plan is less flexible but removes the Python
model stack from request-time execution. The family builder maps checkpoint
weights into TensorRT networks and serializes engines; the family runtime binds
Task inputs and drives those engines through a backend-neutral Engine API.

## Why bundles exist

The `.bundle` file is the boundary between Python build and native runtime. Its
small shared header declares:

```json
{
  "format": 1,
  "family": "qwen",
  "task": "text_generation",
  "backend": "trt"
}
```

Named sections follow the header. They can contain TensorRT plans, tokenizer
data, runtime JSON, preprocessing weights, or other family-owned bytes. Shared
bundle code validates names and bounds but does not interpret section content.

## Family, Task, and backend

These are the three current identities:

| Identity | Example | Meaning |
| --- | --- | --- |
| Family | `qwen`, `whisper`, `flux` | Owns checkpoint matching, build, native runtime, dependencies, and tests. |
| Task | `text_generation`, `transcription`, `image_generation` | User-visible behavior implemented through an abstract C++ interface. |
| Backend | `trt`, `trt_rtx` | Engine implementation selected by the bundle. |

At build time, exactly one dependency-free `support.py` claims the checkpoint
and the core calls only that family's `model.py`. At runtime, the loader opens
only that family's DSO and the named backend from an explicit runtime root.
There is no central model/strategy registry or sibling fallback.

## Reference versus deployment result

Official Hugging Face, Diffusers, NeMo, or other framework execution is the
oracle; the bundle/Task path is the deployment system under test. A successful
build or plausible output is not parity evidence. Family E2E tests align the
exact revision, inputs, precision, generation settings, and model-specific
comparison before accepting a result.

## Self-check

1. Why is a checkpoint not interchangeable with a TensorRT engine?
2. Which header value selects the family DSO, and which names user behavior?
3. What computation does KV cache avoid repeating?

Next, use [Quick Start](quick-start.md), then read
[Architecture Overview](../architecture/overview.md).
