---
title: Nemotron Bolt-74B
---

# Nemotron Bolt-74B

Use the `nemotron_bolt` family for the prepared Bolt-74B target. It supports text and has image and timestamped video interfaces. It does not provide an audio interface. Select ordinary generation by omitting `--execution-variant`; for speculative generation, pass both `--execution-variant lilicorr` and `--companion draft=LOCAL_DIR`.

Build an ordinary bundle from the local prepared target, then run the installed CLI:

```sh
trtmc nemotron_bolt build "$TARGET" --precision bf16 --task vision_language_generation \
  --max-sequence-length 8192 --edge-builder-python "$EDGE_PYTHON" -o bolt.bundle
trtmc run bolt.bundle --prompt 'List the first ten prime numbers, separated by commas.' \
  --use-chat-template true --enable-thinking false --temperature 0 --top-k 1 \
  --top-p 1 --max-new-tokens 96
```

Set `EDGE_PYTHON` to the absolute path of the 0.12.0 builder interpreter when using Edge offload. The runner returns JSON; read its `text` field for the generated answer. On this prompt, the installed Edge route returned the ten primes exactly. The measured image and video answers missed their reference labels; they are executable, not accuracy-qualified.

See the [family recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/nemotron_bolt/README.md) for exact checkpoints, build commands, SDK discovery and the observed media accuracy limitations.
