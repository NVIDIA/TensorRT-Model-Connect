---
title: Text Generation
description: Configure deterministic or sampled generation and related text task commands.
---

Build and inspect an exact text-generation checkpoint, then choose deterministic
or sampled decoding.

```bash
trtmc run model.trtfb \
  --prompt "Summarize TensorRT in one sentence." \
  --max-new-tokens 48 \
  --greedy
```

For sampling, omit `--greedy` and set only the controls required by the
experiment:

```bash
trtmc run model.trtfb \
  --prompt "Write a two-line GPU poem." \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 17
```

| Goal | Controls |
| --- | --- |
| Reproducible smoke test | `--greedy` and a fixed prompt/output bound |
| Reproducible sampling comparison | Fixed `--seed` plus identical temperature/top-k/top-p/min-p |
| Chat formatting | `--chat-template`; inspect the packaged tokenizer/template assets |
| Suppress model thinking mode where supported | `--no-thinking` |
| Language-controlled seq2seq | Source/forced-BOS token IDs required by the exact family contract |

Use [Sampling Reference](../features/sampling.md) for algorithm semantics,
[CLI Reference](../api/cli-reference.md#runtime-commands) for every accepted
option, and the [Text Generation Tutorial](../tutorials/beginner/text-generation.md)
for a progressive lab with exercises.
