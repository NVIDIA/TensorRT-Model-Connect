---
title: Beginner Tutorial - Text Generation
---

import Diagram from '@site/src/components/Diagram';

Complete the [Quick Start](/getting-started/quick-start) before this tutorial.
This page reuses `./qwen3-0.6b.bundle`; it does not build a second newcomer
bundle.

<div className="trtmc-handout-meta">
  <div>
    <strong>Level</strong>
    <span>Beginner</span>
  </div>
  <div>
    <strong>Model</strong>
    <span>`Qwen/Qwen3-0.6B`</span>
  </div>
  <div>
    <strong>Artifact</strong>
    <span>`./qwen3-0.6b.bundle`</span>
  </div>
  <div>
    <strong>Runtime</strong>
    <span>Decoder text generation with KV cache.</span>
  </div>
</div>

<Diagram
  src="/img/diagrams/trtmc-inference-loop.svg"
  alt="Text generation prefill and decode loop with KV-cache reuse"
  caption="Prefill processes the prompt once; decode reuses the KV cache for each new token."
/>

## 1. Read the existing bundle

```bash
trtmc inspect ./qwen3-0.6b.bundle
trtmc inspect ./qwen3-0.6b.bundle --list-engines
```

Connect the output to the runtime:

| Field or section | Meaning |
| --- | --- |
| `family=qwen` | The Python Qwen family built the artifact. |
| `runtime_strategy=qwen_decoder_kv_cache` | The native runtime loads the Qwen decoder implementation. |
| `prefill_engine_plan` | Processes the prompt. |
| `engine_plan` | Produces one token per decode step. |
| `max_cache_length` | Bounds prompt plus generated tokens for this bundle. |
| Tokenizer sections | Convert between text and token IDs. |

## 2. Understand generation

For one request, the runtime:

1. applies the chat template and tokenizes the prompt;
2. runs the prefill engine;
3. stores key/value tensors in the KV cache;
4. runs the decode engine one token at a time;
5. samples each next token from logits; and
6. stops at EOS or `max_new_tokens`.

The KV cache avoids recomputing attention for every earlier token on each
decode step.

## 3. Change one sampling control

Start from the exact run command in Quick Start. Change only one control per
experiment:

| Option | Effect |
| --- | --- |
| `--temperature` | Controls how strongly score differences affect sampling. |
| `--top-k` | Keeps only the highest-scoring candidate tokens. |
| `--top-p` | Keeps the smallest candidate set reaching the probability threshold. |
| `--seed` | Repeats sampling for an unchanged bundle and runtime environment. |
| `--greedy` | Always selects the highest-scoring token instead of sampling. |

Keep `--chat-template` and `--no-thinking` while comparing decoding behavior so
the prompt format does not change at the same time.

## 4. Explain the result

You are done when you can answer:

1. Why are prefill and decode separate engines?
2. What does the KV cache reuse?
3. Which bundle field selects the native Qwen runtime?
4. Why can sampled output change when the software or hardware cohort changes?
5. Which single option would you change for the next experiment?

For exact CLI options, use the [CLI Reference](/api/cli-reference). For parity
and performance work, continue to
[Validation and Benchmarking](/tutorials/advanced/validation-and-benchmarking).

{/* Collaborative review anchor. */}
