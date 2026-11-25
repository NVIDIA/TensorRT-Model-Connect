# Gemma 3n

This family builds `Gemma3nForConditionalGeneration` checkpoints (`model_type: gemma3n`) for the
`images_text_to_text` Task: one or more RGB images and text in a chat conversation, greedy or
sampled text out, through the CLI (`trtmc run BUNDLE --image IMAGE --prompt TEXT`) and the C/C++ SDK.

Validated checkpoint: [`unsloth/gemma-3n-E4B-it`](https://huggingface.co/unsloth/gemma-3n-E4B-it)
at revision `45e9fb1dd0e34db5ff9db1f43a49ac5d8e8b8778` (7.85 B parameters stored as BF16).

## What the bundle contains

- `vision.plan`: the MobileNetV5-300M image encoder (768 x 768 input, multi-scale fusion to a 16 x 16
  grid) and the multimodal embedder that projects its 256 soft tokens into the text space.
- `decoder.plan`: the text decoder with one prefill and one decode optimization profile. It keeps the
  four AltUp streams, the LAuReL blocks, the per-layer embeddings (one token table per layer), the
  Gaussian top-k activation sparsity of the first layers, sliding-window and global attention, and
  the KV sharing of the last layers (they attend to the keys and values of the last earlier layer of
  the same attention type, so only those owners have a cache). Token ids of the vision and audio
  vocabularies use the embedders' hard embeddings, as in the reference model.
- `tokenizer.json`, `runtime.json` and the checkpoint's configuration files.

The runtime renders the chat template, tokenizes, resizes images (antialiased bilinear on 8-bit
pixels, as the reference image processor's torchvision path does) and runs the decode loop. The
build refuses tokenizer pipelines and image-processor settings the runtime does not implement; chat
templates that render a set of probe conversations differently from the runtime (rendered with
Jinja2 as Transformers does); a tokenizer BOS token other than `<bos>`, with which Transformers adds
special tokens to the rendered prompt; and a language-model head that is not tied to the token
embeddings.

## Precision and limits

- `bf16` (default, the stored precision) and `fp32`. `fp16` is refused: decoder activations exceed
  its range. `--quantization fp8` (with `bf16`) quantizes the decoder MLP at build time, see
  [Quantization](#quantization); the default is `none`.
- `max_sequence_length` (default 2048) bounds the prompt plus generated tokens; one image takes
  260 tokens of it.
- Requests carry text and image parts in user/assistant turns with an optional leading system
  message; tool definitions, tool calls and reasoning parts are rejected. Images are RGB, as 8-bit
  or float32 `[0, 1]` pixels (float32 pixels are quantized to 8 bits first).
- Generation settings: `max_new_tokens`, `temperature` (0 is greedy), `top_k`, `top_p`, `seed`.
  Generation stops at `<end_of_turn>` or `<eos>`.
- The audio tower is not built; audio inputs are not part of this Task.

## Quantization

```bash
python -m tensorrt_model_connect build /path/to/gemma-3n-E4B-it --family gemma3n \
  --task images_text_to_text --precision bf16 --quantization fp8 --max-sequence-length 1024 \
  -o gemma-3n-e4b-fp8.bundle
```

No FP8 or NVFP4 checkpoint of Gemma 3n exists, so `--quantization fp8` quantizes the original
checkpoint while it builds the decoder engine (`quantization.py`); FP8 weights are never stored. The
recipe follows the dense-text and VLM guidelines of ModelOpt (language model only, W8A8, static
per-tensor scales, max calibration), narrowed by the measurements below:

| Part | Precision |
| --- | --- |
| Gate, up and down projections of the 35 decoder layers | FP8 W8A8: weight scale `amax / 448` per projection, static input scale, BF16 Dequantize |
| Down projections of layers 0, 1 and 4, gate and up of layer 4 | BF16: calibrated input scales 27.7 to 78.9 times the median of their kind (the limit is 20 times) |
| q, k, v and o projections | BF16: in FP8 they raise the KL divergence by about 40% (0.0124-0.0128 to 0.0175-0.0180) for 1% of decode time |
| MobileNetV5 tower, embedders, token and per-layer embeddings, tied head, AltUp, LAuReL, per-layer input projections, norms, attention core | As in the unquantized build |

The input scales (`fp8_activation_scales.json`) are `amax / 448` from ModelOpt `FP8_DEFAULT_CFG` on
the BF16 model over 384 chat samples (336 COCO validation images with an instruction, 48 text-only
turns, each followed by a teacher-forced news or Wikipedia text of up to 700 tokens). Rerun
`calibrate_scales.py` for another checkpoint: an entry is keyed by the decoder config and also stores
the first 8 values of the final norm weight, so another checkpoint of the same config is refused, as
are a checkpoint that is already quantized and any precision but `bf16`. FP8 builds turn the fused
gate/up (dual GEMM) kernel off: with it the decode step is 7% slower on B200 and equal on B300.

Measured with TensorRT 11.4 on B200 and B300, `max_sequence_length` 1024, batch 1, the engines run
by a Python driver that feeds them like the runtime (all variants of a GPU built and timed in one job,
medians of 9 interleaved rounds). A request is the vision engine, a 275-row prompt (one image) and 64
decode steps:

| Engine | Decoder plan | Decode step B200 / B300 | Prompt B200 / B300 | Request B200 / B300 |
| --- | --- | --- | --- | --- |
| BF16 | 13128 MiB | 6.962 / 6.711 ms | 14.02 / 13.51 ms | 468.8 / 451.4 ms |
| FP8 | 9906 MiB (-25%) | 6.527 / 6.247 ms (1.07x / 1.07x) | 13.06 / 12.52 ms (1.07x / 1.08x) | 440.0 / 420.9 ms (1.07x / 1.07x) |

In short: the FP8 speed gain is small (1.07x) and the main gain is size: the bundle shrinks from
14.41 GB to 11.06 GB (-23%). The speed gain is small because the decode step is
not bound by reading weights: it runs about 1030 kernels, mostly the per-layer elementwise work of
AltUp, LAuReL, the per-layer inputs and the norms (a CUDA-graph replay of the step takes 6.4 ms
against 6.8 ms of plain enqueue on B300), and FP8 shortens only the MLP GEMMs. TensorRT 11.1 builds
the FP8 engine too (B300: decode 1.02x, KL 0.0126).

Quality against the original model: transformers `Gemma3nForConditionalGeneration` in FP32 (eager
attention, TF32 off), KL(reference || engine) per token and top-1 agreement on 5376 held-out
teacher-forced tokens (24 COCO images with an instruction and a 224-token news or Wikipedia answer;
none used for calibration):

| Engine | KL mean | KL p99 | Top-1 |
| --- | --- | --- | --- |
| transformers BF16 | 0.0017 | 0.015 | 0.977 |
| BF16 engine | 0.0009 | 0.010 to 0.011 | 0.985 to 0.987 |
| FP8 (shipped) | 0.0124 to 0.0128 | 0.13 to 0.15 | 0.950 to 0.952 |
| FP8, attention output also FP8 | 0.0152 | 0.17 to 0.19 | 0.945 to 0.947 |
| FP8, all seven projections | 0.0175 to 0.0180 | 0.18 to 0.20 | 0.944 |

Ranges are B200 to B300. Two 700-token answers (cache rows up to 980) give KL 0.0120 to 0.0127 for
the shipped engine. No logit was non-finite, also for 14 greedy captions and for noise, black and
white images with greedy runs up to the full cache.

NVFP4 was measured and is not shipped: it needs a KL of at most about twice that of FP8, and the best
layout is about 8 times. NVFP4 used `add_dynamic_quantize`, per-16 E4M3 block scales,
`weight_scale_2 = amax / (6 * 448)` and the FP8 input scale over 6; every layout builds and runs on
TensorRT 11.4 without a non-finite value:

| NVFP4 layout (rest FP8, outliers BF16) | Decoder plan | Decode step vs FP8 (shipped), B200 / B300 | KL mean | Top-1 |
| --- | --- | --- | --- | --- |
| Gate and up NVFP4 | 8679 MiB | 1.03x / 1.06x | 0.098 to 0.102 | 0.873 to 0.877 |
| Gate, up and down NVFP4, layers 0, 1, 33, 34 FP8 | 8324 MiB | 1.02x / 1.04x | 0.125 to 0.127 | 0.858 to 0.864 |
| Gate, up and down NVFP4 | 8185 MiB | 1.02x / 1.04x | 0.131 to 0.137 | 0.854 to 0.855 |
| All seven projections NVFP4 | 8081 MiB | 0.99x / 1.01x | 0.203 to 0.207 | 0.813 to 0.814 |

## Validation

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest families/gemma3n/tests/test_support.py
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest families/gemma3n/tests \
  --e2e-model gemma3n -vv
```

The E2E builds each manifest's bundle and runs five conversations (one image, a system prompt, two
images, a multi-turn follow-up and float32 pixels) through the C++ SDK consumer, one of them through
the C SDK consumer and the CLI, which must return the same tokens and text. Against the Transformers
model in FP32, the native prompt token ids of every conversation (on the checkpoint's tokenizer) must
be the reference processor's, and the native text the reference tokenizer's decoding of the native
tokens; FP32 bundles must reproduce the reference's greedy tokens exactly; BF16 bundles are checked teacher-forced against
the FP32 reference within the logit margin in `tests/thresholds/`, which is the reference model's own
BF16 error on the gap between its two best tokens along these conversations. The FP8 bundle uses the
same check with a 1.5-logit margin: its largest margin was 0.014 on B200 and 0.150 on B300 (the BF16
bundle's 0.150 there too), and a build with every activation scale multiplied by 0.1 reaches 8.2. Each conversation's checks
are recorded as test evidence, so the E2E report shows the first conversation's image, prompt and texts,
and each recorded comparison check with its value and limit. `test_gemma3n_processing`
(CTest) checks the tokenizer, the image resize and the chat rendering against goldens recorded from
the reference stack, and invariants of the sampler (greedy first maximum, top-k 1, seeded
reproducibility, a tiny nucleus). `tests/test_quantization.py` covers the FP8 recipe, the scale
file, the rejected requests and a tiny FP8 decoder engine.

The image resize is adapted from PyTorch (BSD-3-Clause), which derives it from Pillow-SIMD
(MIT-CMU); see `NOTICE`.
