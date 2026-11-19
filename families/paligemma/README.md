# PaliGemma

This family owns `PaliGemmaForConditionalGeneration` checkpoints whose language
model is the first-generation Gemma decoder (`model_type: paligemma` with a
`gemma` text config). PaliGemma 2 (`gemma2` text config) is not claimed. The
E2E checkpoint is `google/paligemma-3b-pt-224` at revision
`35e4f46485b4d07967e7e9935bc3786aad50687c`; it is gated, so downloading it needs
a Hugging Face token whose account has accepted the Gemma license.

## Inventory and boundaries

| Input | Output | Task | Execution |
| --- | --- | --- | --- |
| One or more RGB images, then prefix text | Text | `images_text_to_text` | Greedy autoregressive |

The family runs the complete SigLIP vision tower, the linear projector and the
Gemma decoder. Supported precisions are `fp32` and `bf16` (default); the
checkpoint stores FP32 weights. `fp16` is rejected: it is not qualified for
the Gemma decoder. The family does not implement sampling, beam search,
batching, tensor or context parallelism, quantized checkpoints, a chat
template, text-only prompts or images placed after text.

## Build

```sh
python -m tensorrt_model_connect build /path/to/paligemma-3b-pt-224 \
  --family paligemma --task images_text_to_text --precision bf16 \
  --max-sequence-length 1024 -o paligemma-3b-pt-224-bf16.bundle
```

`--max-sequence-length` (default 1024) is the decoder's cache capacity: the
prompt (256 tokens per 224-pixel image, BOS, the prefix and its newline) plus
all generated tokens but the last. Longer requests fail instead of being
shortened. The bundle holds two TensorRT engines built directly with the
TensorRT Python API (`vision.plan`, `decoder.plan`), the FP32 token-embedding
table the host gathers from (`embeddings.f32`), the flattened SentencePiece
tokenizer (`tokenizer.bpe`) and `runtime.json`.

Building needs `sentencepiece` and `protobuf` to read `tokenizer.model`, and
`safetensors` with `torch` to read weights (see `requirements.txt`). Both
transformers 4.x and 5.x checkpoint layouts are accepted.

## Runtime behavior

The runtime follows the transformers 5.2.0 model and the checkpoint's saved
processor, `SiglipImageProcessor`:

- Each image is squashed onto the 224x224 square with Pillow 13's bicubic
  filter, reproduced exactly, including its choice of which 8-bit pass runs
  first (the height pass when the height shrinks by more than twice as much as
  the width; Pillow 12 always starts with the width). The image is then
  rescaled by 1/255 and normalized with mean and std 0.5.
  No aspect ratio is preserved and nothing is cropped. When torchvision is
  installed, `AutoProcessor.from_pretrained` in transformers 5.2.0 selects the
  fast image processor instead, which resamples with torchvision and can
  differ slightly; the E2E reference therefore loads the processor with
  `use_fast=False`.
- The prompt is `<image>` x 256 per image, `<bos>`, the prefix text and `\n`.
  Text parts are concatenated. The reference splits every added token out of
  the text first; SentencePiece does the same only for its user-defined
  pieces, so a prefix that contains any other added token (for example
  `<bos>`, `<eos>` or `<image>`) is rejected rather than tokenized
  differently.
- The whole prompt attends bidirectionally; generated tokens attend causally.
  RoPE positions start at 1. Image features are divided by the square root of
  the Gemma width and, like text embeddings, multiplied by it again in the
  decoder.
- Decoding is greedy and stops at the checkpoint's EOS ids or after
  `max_new_tokens` (default 64). Special tokens, as transformers 5.2.0 loads
  them from `tokenizer_config.json` (or `tokenizer.json` without it), are
  skipped in the decoded text; location and segmentation tokens (`<loc0000>`,
  `<seg000>`) are kept.

## Validation

```sh
TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/paligemma/tests --e2e-model paligemma -vv
```

Build the CMake targets `trtmc`, `trtmc_backend_trt`, `trtmc_model_paligemma`,
`test_paligemma_sdk_cpp` and `paligemma_tokenizer_check` first. The E2E builds
the FP32 and BF16 bundles, runs five requests through the public
`images_text_to_text` Task and compares them with the transformers FP32
reference (eager attention, TF32 off, slow image processor). The requests use
the repository's shared test photograph, `tests/data/test_img.jpeg` (see
`ASSET_LICENSES.md`), and a tall synthetic image: a caption, a VQA question
and a detection on the photograph, a VQA question on the synthetic image, and
a caption of both images together. The FP32 bundle must reproduce the
reference tokens exactly. For the BF16 bundle the FP32 reference is forced
through the native tokens: every native token must lie within 1.0 logit of the
reference's top logit (the official model run in BF16 departs from its FP32
argmax by up to 0.548 logit on 41 image / prompt pairs), and the native text
must equal the reference processor's decoding of the native tokens; a
free-running BF16 text can leave the FP32 text after such a near tie. Both
also check the native prompt ids and the decoding of the reference tokens
with the tokenizer packaged in the bundle. With
`TRTMC_E2E_ARTIFACT_DIR` set, each case's evidence keeps the input images, the
native and reference outputs and one reference comparison per request.

`tests/test_graph.py` compares the FP32 and BF16 engines with transformers on a
seeded tiny random checkpoint (no download), and `tests/test_model.py` plus the CTest
`paligemma_runtime` cover configuration, checkpoint layouts, tokenizer
flattening and decoding, prompt assembly and the Pillow resize on CPU.
