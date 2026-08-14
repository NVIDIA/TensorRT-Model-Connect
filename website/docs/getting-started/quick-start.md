---
title: Your First NLP Inference
---

Complete [System Requirements](environment-and-repro.md), then use either
[Installation](installation.md) or [Build from Source](source-build.md).

## 1. Check the CLI

```bash
trtmc version
```

Expected signals include:

```text
trtmc 0.1.0
TRT support: yes
```

## 2. Build Qwen

```bash
trtmc build Qwen/Qwen3-0.6B \
  --precision bf16 \
  --max-cache-length 16384 \
  --output qwen3-0.6b.bundle
```

The bounded cache profile is intended for the first portable native-attention
build. The first build may download model files and compile TensorRT engines.

## 3. Inspect the bundle

```bash
trtmc inspect ./qwen3-0.6b.bundle
trtmc inspect ./qwen3-0.6b.bundle --list-engines
```

For this journey, confirm only the `qwen` family, `qwen_decoder_kv_cache`
runtime strategy, BF16 precision, the configured cache length, and two listed
engine plans. Generic fields are not used by this text-generation path.

## 4. Run Qwen

```bash
trtmc run ./qwen3-0.6b.bundle \
  --prompt "What is the capital of France? Answer in one word." \
  --chat-template \
  --no-thinking \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-k 20 \
  --top-p 0.8 \
  --seed 42
```

Success returns `Paris` and stops without a fatal build, load, or inference
error. If it does not, keep the first error and use
[First-run Troubleshooting](troubleshooting.md).

Continue with [Learning Path](../learning-path.md) or choose another model from
[Model Support](../models-recipes/overview.md).

{/* Collaborative review anchor. */}
