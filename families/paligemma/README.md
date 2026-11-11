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

- Each image is squashed onto the 224x224 square with Pillow's bicubic filter
  (reproduced exactly), rescaled by 1/255 and normalized with mean and std 0.5.
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
the FP32 and BF16 bundles, runs five requests (captioning, VQA, detection, a
tall image and a two-image prompt) through the public `images_text_to_text`
Task, and compares them with the transformers FP32 reference (eager attention,
TF32 off, slow image processor). The FP32 bundle must reproduce the reference
tokens exactly; the BF16 bundle is held to a normalized text edit distance.
Both also check the native prompt ids and decoded text against the reference
processor with the tokenizer packaged in the bundle.

`tests/test_graph.py` compares both FP32 engines with transformers on a seeded
tiny random checkpoint (no download), and `tests/test_model.py` plus the CTest
`paligemma_runtime` cover configuration, checkpoint layouts, tokenizer
flattening and decoding, prompt assembly and the Pillow resize on CPU.
