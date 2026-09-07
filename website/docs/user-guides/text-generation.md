---
title: Text Generation
description: Configure deterministic or sampled generation and related text task commands.
---

Inspect an exact text-generation bundle, then execute it with the native
runtime directory:

```bash
trtmc inspect model.bundle

trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Summarize TensorRT in one sentence." \
  --max-new-tokens 48 \
  --temperature 0 \
  --top-k 1
```

For sampling, set only the controls needed by the experiment:

```bash
trtmc run model.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Write a two-line GPU poem." \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 17
```

| Goal | Controls |
| --- | --- |
| Deterministic smoke test | `--temperature 0`, `--top-k 1`, fixed prompt and output bound |
| Reproducible sampling comparison | Fixed seed and identical sampling controls |
| Packaged chat formatting | `--use-chat-template true` |
| Disable a supported thinking mode | `--enable-thinking false` |
| Language-controlled seq2seq | Family-supported source and forced-BOS token IDs |

The family implements tokenization, prefill/decode, cache management,
sampling, and stopping behind `ITextGeneration`. Shared core does not impose
one language-model pipeline on every family.

## LFM2 model-card-style request

```bash
python -m tensorrt_model_connect build LiquidAI/LFM2-350M \
  --precision bf16 \
  --max-sequence-length 32768 \
  --output lfm2-350m.bundle

trtmc run lfm2-350m.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "What is C. elegans?" \
  --use-chat-template true \
  --max-new-tokens 512 \
  --temperature 0.3 \
  --min-p 0.15 \
  --top-k 50 \
  --repetition-penalty 1.05
```

The CLI returns one JSON result per invocation. To run several prompts through
the same loaded task, build an application against the public C++ Task API;
examples and apps remain one-way consumers and do not become family
dependencies.
