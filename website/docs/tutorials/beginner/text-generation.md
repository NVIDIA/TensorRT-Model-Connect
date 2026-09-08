---
title: Beginner Tutorial - Text Generation
---

# Text generation

Complete the [Quick Start](/getting-started/quick-start) first. This tutorial
reuses `./gpt2.bundle`.

## Inspect and run

```bash
trtmc inspect ./gpt2.bundle

trtmc run ./gpt2.bundle \
  --prompt "Explain why KV caches help decoding." \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 1234
```

The runtime tokenizes the prompt, runs prefill once, then repeatedly runs
decode while reusing key/value tensors. Generation stops at EOS or the
requested maximum.

## Change one control at a time

| Option | Effect |
| --- | --- |
| `--temperature` | Rescales logits before sampling. |
| `--top-k` | Limits sampling to the highest-scoring candidates. |
| `--top-p` | Limits sampling to a cumulative probability mass. |
| `--min-p` | Removes candidates far below the best probability. |
| `--seed` | Controls the sampling RNG for the same software and target. |
| `--repetition-penalty` | Adjusts scores for tokens already generated. |

The current CLI does not provide `--greedy`; configure deterministic selection
with the supported sampling controls for the installed version. Chat-template
and reasoning behavior use the explicit boolean options shown in the
[CLI Reference](/api/cli-reference), such as `--use-chat-template true|false`
and `--enable-thinking true|false`.

You are done when you can explain why prefill and decode are distinct phases,
what the KV cache reuses, and why identical sampling flags still require the
same bundle, runtime, and target environment for a reproducible comparison.
