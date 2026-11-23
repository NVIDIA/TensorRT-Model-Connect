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

The native fallback accepts at most `min(1024, max_input_length)` visual tokens
per request, shared across all images and video frames. A larger language
context does not increase this visual limit. If provider-sized media would
exceed it, the runtime reports the limit rather than silently resizing or
dropping content.

An earlier resident Edge run matched the text, image, and ordered-video
references. The latest native run matched text and image, but answered
“red panda, panda” instead of “red panda, giant panda” for the video. An
earlier native run matched all three. Row-cached Edge also missed the last
animal’s full name. Both routes remain executable; neither is qualified for
ordered video.
