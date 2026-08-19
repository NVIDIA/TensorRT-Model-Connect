---
title: Text Generation
description: Configure deterministic or sampled generation and related text task commands.
---

Build and inspect an exact text-generation checkpoint, then choose deterministic
or sampled decoding.

```bash
trtmc run model.bundle \
  --prompt "Summarize TensorRT in one sentence." \
  --max-new-tokens 48 \
  --greedy
```

For sampling, omit `--greedy` and set only the controls required by the
experiment:

```bash
trtmc run model.bundle \
  --prompt "Write a two-line GPU poem." \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 17
```

| Goal | Controls |
| --- | --- |
| Reproducible smoke test | `--greedy` and a fixed prompt/output bound |
| Reproducible sampling comparison | Fixed `--seed` plus identical temperature/top-k/top-p/min-p/repetition penalty |
| Chat formatting | `--chat-template`; inspect the packaged tokenizer/template assets |
| Suppress model thinking mode where supported | `--no-thinking` |
| Language-controlled seq2seq | Source/forced-BOS token IDs required by the exact family contract |

Use [Sampling Reference](../features/sampling.md) for algorithm semantics,
[CLI Reference](../api/cli-reference.md#runtime-commands) for every accepted
option, and the [Text Generation Tutorial](../tutorials/beginner/text-generation.md)
for a progressive lab with exercises.

## LFM2 model-card equivalent

Build the dense Liquid AI checkpoint directly. The family defaults to BF16 and
the model card's 32,768-token context capacity; pass `--precision fp16` for the
issue #928 FP16 qualification route.

```bash
trtmc build LiquidAI/LFM2-350M -o lfm2-350m.bundle

trtmc run lfm2-350m.bundle \
  --prompt "What is C. elegans?" \
  --chat-template \
  --max-new-tokens 512 \
  --temperature 0.3 \
  --min-p 0.15 \
  --top-k 50 \
  --repetition-penalty 1.05
```

`--chat-template` matches the model card's single-user
`apply_chat_template(..., add_generation_prompt=True)` example. Omit it for
the raw-prompt form shown in the vLLM example. The model card does not spell
out `top_k`, but Transformers sampling inherits `top_k=50`; the command makes
that hidden default explicit.

The vLLM card example's raw-prompt batch has a sequential native equivalent:

```bash
printf '%s\n' \
  "What is C. elegans?" \
  "Say hi in JSON format" \
  "Define AI in Spanish" > lfm2-prompts.txt

trtmc run lfm2-350m.bundle \
  --prompts-file lfm2-prompts.txt \
  --max-new-tokens 16 \
  --temperature 0.3 \
  --min-p 0.15 \
  --top-k 0 \
  --repetition-penalty 1.05
```

Each JSONL result includes its input prompt, generated text, and token IDs;
requests execute independently through the same loaded pure C++ pipeline.
Here `top_k=0` and 16 output tokens mirror the vLLM example's defaults; they
are intentionally different from the Transformers example's hidden
`top_k=50` default above.

The packaged tokenizer preserves LFM2's system, assistant, tool-list,
tool-call, and tool-response special tokens. The scalar CLI chat helper creates
one user turn. For multi-turn or tool-use examples, render the documented
ChatML-like string first and pass it as the raw prompt without
`--chat-template`; the native runtime tokenizes and continues that exact
string. OpenAI-compatible HTTP serving, vLLM/SGLang objects, Docker Model
Runner, GGUF/llama.cpp, and fine-tuning notebooks are external wrappers or
alternate formats rather than TensorRT Model Connect runtime APIs.

{/* Collaborative review anchor: batch 2. */}
