---
title: Qwen3.8 Flash-Next NVFP4 with Edge-LLM
---

Use the ordinary prepared `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint
without a draft. The `fp16` option selects activation precision; it does not
convert the prepared NVFP4 expert weights or the FP8 lexical table.

```sh
trtmc qwen3_8_flash_next build nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --task vision_language_generation --precision fp16 --max-sequence-length 8192 \
  --edge-builder-python /path/to/edge-env/bin/python -o flash-next.bundle
trtmc run flash-next.bundle --prompt 'List the first ten prime numbers.' \
  --use-chat-template true --enable-thinking false --max-new-tokens 64
```

The offload path needs a Model Connect runtime with the matching Edge-LLM
0.12.0 SDK and a builder interpreter with Edge-LLM 0.12.0 installed.
The default Edge lookup table is GPU-resident. Add `--edge-ple-residency cached`
at build time for the released row-cached policy: Edge keeps the table on the host
and stages the rows it needs. Edge documents this policy for Jetson Thor. This
option changes storage, not the model weights or decoding algorithm.

Edge preparation failures warn and select the native TensorRT builder; inference
errors are reported rather than retried on a different backend. Both routes
build self-contained bundles. Budget for the original checkpoint, the bundle,
and temporary build and inference files.

The resident policy and native fallback passed the family’s three focused text,
image, and ordered-video checks. The cached policy built and ran but missed the
last animal’s full name in the video check. Do not interpret successful cached
execution as equivalent accuracy.
