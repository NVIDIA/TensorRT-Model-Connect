---
title: Text Generation
description: Configure deterministic or sampled generation and related text tasks.
---

Build and inspect an exact text checkpoint, then call the native Task:

```bash
trtmc run model.bundle \
  --prompt "Summarize TensorRT in one sentence." \
  --max-new-tokens 48 \
  --temperature 0 \
  --top-k 1
```

For reproducible stochastic sampling, fix every sampling input:

```bash
trtmc run model.bundle \
  --prompt "Write a two-line GPU poem." \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --min-p 0.0 \
  --repetition-penalty 1.0 \
  --seed 17
```

| Goal | Current controls |
| --- | --- |
| Greedy-like deterministic decoding | `--temperature 0 --top-k 1` |
| Seeded sampling | Fixed seed plus identical temperature/top-k/top-p/min-p/repetition penalty |
| Chat formatting | `--use-chat-template true` when supported by the family |
| Thinking behavior | `--enable-thinking true\|false` when supported |
| Translation framing | Source-language and forced-BOS token IDs for the exact family contract |
| Dynamic LoRA | Paired `--lora-adapter PATH --lora-adapter-id ID` for a family implementing the interface |

The current CLI accepts one prompt per `run`; it has no `--greedy`,
`--prompts-file`, `--chat-template`, or `--no-thinking` compatibility aliases.
Unknown or family-unsupported values fail explicitly.

Use [Sampling](../features/sampling.md) for algorithm semantics and the
[CLI Reference](../api/cli-reference.md) for exact options.
