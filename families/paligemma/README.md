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
the Gemma decoder. `--quantization fp8` (with `bf16`) quantizes part of the
decoder at build time (see [Quantization](#quantization)); the default is
`none`. The family does not implement sampling, beam search, batching, tensor
or context parallelism, quantized checkpoints, a chat template, text-only
prompts or images placed after text.

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

## Quantization

```sh
python -m tensorrt_model_connect build /path/to/paligemma-3b-pt-224 \
  --family paligemma --task images_text_to_text --precision bf16 \
  --quantization fp8 --max-sequence-length 1024 -o paligemma-3b-pt-224-fp8.bundle
```

No FP8 or NVFP4 checkpoint of PaliGemma exists, so `--quantization fp8`
quantizes the original checkpoint while it builds the decoder engine
(`quantization.py`); FP8 weights are never stored. The recipe, from the
dense-text and VLM guidelines of ModelOpt (language model only, static
per-tensor scales, max calibration) narrowed by the measurements below:

| Part | Precision |
| --- | --- |
| Attention output, gate, up and down projections of all 18 decoder layers | FP8 W8A8: weight scale `amax / 448` per projection, static input scale |
| Down projection of layer 17 | BF16: its calibrated input scale is 28.8 times the median of its kind (the limit is 20 times) |
| q, k and v projections | BF16: in FP8 they raise the KL divergence by 40% (0.0103 to 0.0142 or more) and shorten the decode step by only about 1% |
| SigLIP tower, projector, token embedding and tied head, norms, attention core | BF16 / FP32 as in the unquantized build; the vision engine is 0.7% of a request |

The input scales (`fp8_activation_scales.json`) are `amax / 448` from ModelOpt
`FP8_DEFAULT_CFG` on the BF16 model, over 384 COCO validation images, each with
a task prefix and a teacher-forced suffix (news and Wikipedia text up to 700
tokens). Rerun `calibrate_scales.py` for another checkpoint: an entry is keyed
by the decoder config and also stores the first 8 values of the final norm
weight, so another checkpoint of the same config (448 or 896 pixels, mix) is
refused, as are a checkpoint that is already quantized and any precision but
`bf16`. The Q/DQ pairs use a BF16 Dequantize; no non-finite
value was seen.

Measured with TensorRT 11.4 on B200 and B300, `max_sequence_length` 1024,
batch 1, the engines run by a Python driver that feeds them like the runtime
(all variants of a GPU built and timed in one job, medians of 11 interleaved
rounds). A request is the vision engine, a 260-row prompt and 64 decode steps:

| Engine | Decoder plan | Decode step B200 / B300 | Prompt B200 / B300 | Request B200 / B300 |
| --- | --- | --- | --- | --- |
| BF16 | 4788 MiB | 3.114 / 3.048 ms | 3.436 / 3.353 ms | 208.6 / 203.2 ms |
| FP8 | 3025 MiB (-37%) | 2.171 / 2.056 ms (1.43x / 1.48x) | 2.939 / 2.833 ms (1.17x / 1.18x) | 147.5 / 139.5 ms (1.41x / 1.46x) |

The bundle shrinks from 7.96 GB to 6.11 GB (-23%; the FP32 embedding table and
the vision engine are unchanged). FP8 builds turn the fused gate/up (dual
GEMM) kernel off: TensorRT 11.1 fails to compile it for some configs, and on
11.4 the build without it is as fast (B300) or 7% faster (B200). TensorRT 11.1
builds the FP8 engine too (B300: decode 1.27x to 1.43x over three jobs, KL
0.0103).

Quality against the original FP32 model in transformers (eager attention):
KL(reference || engine) per token and top-1 agreement on 5376 held-out
teacher-forced tokens (24 COCO images with prefixes and 224-token news or
Wikipedia suffixes; none used for calibration):

| Engine | KL mean | KL p99 | Top-1 |
| --- | --- | --- | --- |
| transformers BF16 | 0.0011 | 0.0075 | 0.975 |
| BF16 engine | 0.0010 | 0.0065 | 0.978 |
| FP8 (shipped) | 0.0103 | 0.065 | 0.944 |
| FP8, q/k/v also FP8 | 0.0142 to 0.0145 | 0.082 to 0.091 | 0.925 to 0.927 |
| FP8, MLP only | 0.0083 to 0.0086 | 0.055 to 0.058 | 0.945 to 0.946 |

Ranges are B200 to B300. Two 700-token suffixes (cache rows up to 960) give
KL 0.0081 for the shipped engine. No logit and no K/V row was non-finite, also
for noise, black and white images with 700-token greedy runs up to the full
cache.

NVFP4 was measured and is not shipped: it needs a KL of at most about twice
that of FP8, and the best layout is 4.4 times. Its decode step is 1.06x to
1.12x that of FP8 when both keep the fused gate/up kernel (same job), and
about the same as the shipped FP8 build. NVFP4 uses `add_dynamic_quantize`,
per-16 E4M3 block scales, `weight_scale_2 = amax / (6 * 448)` and the FP8
input scale over 6:

| NVFP4 layout | Decoder plan | Decode step vs FP8, B200 / B300 | KL mean | Top-1 |
| --- | --- | --- | --- | --- |
| Gate, up and down NVFP4; q, k, v, o FP8 | 2191 MiB | 1.12x / 1.06x | 0.0866 | 0.842 |
| Gate and up NVFP4 except layers 0, 1, 16, 17 (FP8); q, k, v, o, down FP8 | 2540 MiB | 1.09x / 1.07x | 0.0455 to 0.0459 | 0.879 to 0.882 |
| All seven projections NVFP4 | 2120 MiB | 1.07x / 1.05x (vs FP8 with q/k/v, other job) | 0.137 | 0.804 |

## Validation

```sh
TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/paligemma/tests --e2e-model paligemma -vv
```

Build the CMake targets `trtmc`, `trtmc_backend_trt`, `trtmc_model_paligemma`,
`test_paligemma_sdk_cpp` and `paligemma_tokenizer_check` first. The E2E builds
the FP32, BF16 and FP8 bundles, runs five requests through the public
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
free-running BF16 text can leave the FP32 text after such a near tie. The FP8
bundle uses the same check with a 2.5-logit tolerance: its largest gap was
1.27 logit on B200 and B300, and a build with every activation scale
multiplied by 0.1 fails it. All cases also check the native prompt ids and
the decoding of the reference tokens with the tokenizer packaged in the
bundle. With
`TRTMC_E2E_ARTIFACT_DIR` set, each case's evidence keeps the input images, the
native and reference outputs and one reference comparison per request.

`tests/test_graph.py` compares the FP32 and BF16 engines with transformers on a
seeded tiny random checkpoint (no download), and `tests/test_model.py` plus the CTest
`paligemma_runtime` cover configuration, checkpoint layouts, tokenizer
flattening and decoding, prompt assembly and the Pillow resize on CPU.
`tests/test_quantization.py` covers the FP8 recipe, the scale file, the
rejected requests and a tiny FP8 decoder engine.
