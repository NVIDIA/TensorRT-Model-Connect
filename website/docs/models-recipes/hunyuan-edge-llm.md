---
title: Hunyuan translation with Edge-LLM
---

Hunyuan can use the Edge builder and C++ runtime for an original-weight
translation checkpoint. When Model Connect is built with the optional 0.12.0
SDK, pass the Python interpreter containing the matching Edge builder. It can
be a PyPI or source installation; it does not replace the native SDK.

An installed Model Connect wheel uses its own native Edge SDK, not a
different SDK on `CMAKE_PREFIX_PATH`. To switch releases, use a matching
wheel or a standalone Model Connect builder and runner together.

```sh
trtmc hunyuan build tencent/Hy-MT2-1.8B --precision fp16 \
  --max-sequence-length 1024 --edge-builder-python /path/to/edge-env/bin/python \
  -o translation.bundle
trtmc run translation.bundle --prompt 'Translate into English: 今天天气真好。' \
  --use-chat-template true --temperature 0 --max-new-tokens 128
```

If an eligible Edge build fails, the family warns and retries native TensorRT
with the same request. A runtime error is reported, not silently retried. See
the [family recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/hunyuan/edge_llm/README.md)
for the validated checkpoints, precision, capacity, and independent translation
checks. Older SDK results do not qualify a 0.12 profile.
