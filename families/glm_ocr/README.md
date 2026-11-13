# GLM-OCR

This family runs `zai-org/GLM-OCR` (`model_type: glm_ocr`, `GlmOcrForConditionalGeneration`)
on the `images_text_to_text` Task: one or more page images plus a text prompt in, the
recognized text out. The E2E cases pin revision `2e85a62840ccac27daa451df36c736c4636b8628`.

| Checkpoint | Inputs | Output | Execution |
| --- | --- | --- | --- |
| `zai-org/GLM-OCR` | Image(s) and text, optionally after system and earlier turns | Text | Autoregressive, greedy |

The family runs the complete model: the 24-block vision tower with its 2x2 downsample and
patch merger, the multimodal rotary positions, and the 16-layer GLM decoder. The
checkpoint's multi-token-prediction layer (decoder layer 16) is not on the generation path
and is not built. Video input, sampling, batching, tensor parallelism and quantized builds
are not offered.

## Build

```sh
python -m tensorrt_model_connect build /path/to/GLM-OCR \
  --family glm_ocr --task images_text_to_text --precision bf16 -o glm-ocr.bundle
```

`bf16` (the default) keeps the checkpoint's BF16 weights; `fp32` builds the same graphs in
FP32 with TF32 disabled. A bundle holds three engines (vision, decoder prefill, one-token
decoder), the BF16 token-embedding table, the checkpoint's `tokenizer.json` and
`runtime.json`.

* `--max-sequence-length` sets the key/value cache: prompt (including image tokens) plus
  generated tokens. Default 16384.
* `--image-height` / `--image-width` (together) declare the largest input image, in pixels.
  The vision engine is sized for every image up to that size after the processor's resize,
  including the enlargement of small images. Default: the processor's own limit (9,633,792
  pixels over two frames, i.e. 24,576 vision patches). An image that needs more patches is
  refused with a message, never shrunk.

## Run

```sh
trtmc run glm-ocr.bundle --image page.png --prompt "Text Recognition:" --max-new-tokens 1024
```

The prompts the model card documents are `Text Recognition:`, `Formula Recognition:`,
`Table Recognition:` and JSON-schema information extraction. Generation is greedy. Task
options: `max_new_tokens` (without it generation stops at an end-of-sequence token or a
full cache), and `temperature`, `top_k`, `top_p` only with values that keep decoding greedy
(`top_k` 1, the default, or `temperature` 0), as the shared vision profile sends them;
sampling requests are refused. Template controls such as `enable_thinking` are not offered.

The native runtime reproduces the checkpoint's host-side processing: the chat template
(system, user and assistant messages; images in user messages), the tokenizer
(`tokenizer.json`'s byte-level BPE with its pre-tokenizer regex and added tokens), and
the torchvision `Glm46VImageProcessor`: `smart_resize`, PyTorch's antialiased bicubic
resize of 8-bit images, CLIP normalization and merge-window patch order. Images are RGB
(UInt8, or Float32 values on the 1/255 grid).

## Validation

`tests/test_e2e.py` builds each manifest's bundle, runs the CLI and runs the official
Transformers model on the same pixels:

* `fp32`: tokens and text identical to the official FP32 model.
* `bf16`: teacher-forced against the official FP32 model, each token within 2.0 logits of
  the FP32 maximum (the calibration is in the test).

Its host-parity tests compare the native preprocessing (bit-exact), tokenizer, pre-tokenizer
(every Unicode scalar and random text), prompt rendering and rotary positions with the
processor and the tokenizers library. They run the host probe of `test_glm_ocr_runtime`, the
family's CTest executable, which without arguments covers the same host steps without a
checkpoint (`tests/cpp/test_runtime.cpp`).

```sh
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/glm_ocr/tests --e2e-model glm_ocr -v
```

## Third-party notices

`NOTICE` lists the adapted code: the antialiased bicubic resize in `runtime/image.cpp`
(PyTorch v2.9.1, BSD-3-Clause, derived from Pillow-SIMD, MIT-CMU License) and the
processor, prompt, position and tokenizer behaviors reproduced from Hugging Face
Transformers v5.22.2 and tokenizers (Apache-2.0). It is installed with this family,
next to `UNICODE_LICENSE.txt`.

## Unicode data

The pre-tokenizer's letter, number and white-space classes (`runtime/unicode_tables.h`)
are generated from the Unicode 16.0
[UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt) (SHA-256
`ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`), redistributed under the
Unicode License v3 in `UNICODE_LICENSE.txt`, which is installed with this family. They
match the tokenizers library's regex for every Unicode scalar (a host-parity test).
Regenerate with:

```sh
python families/glm_ocr/tools/generate_unicode_tables.py \
  UnicodeData.txt families/glm_ocr/runtime/unicode_tables.h
```
