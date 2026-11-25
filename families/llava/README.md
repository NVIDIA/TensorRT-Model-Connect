# LLaVA 1.5

This family builds `llava-hf/llava-1.5-7b-hf` into native TensorRT engines for
`vision_language_generation`. The checked testcase pins Hugging Face revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.

The supported path uses FP16, one GPU, one image per request, and the
checkpoint's 336×336 CLIP input. `--quantization fp8` quantizes the Llama
decoder at build time (see [Quantization](#quantization)); the default is
`none`. CLIP's penultimate patch features pass
through the GELU projector and replace 576 image tokens in the Llama prompt.
Image preprocessing reproduces Pillow bicubic resize and center crop. Prompts
are plain user questions; the runtime adds the checkpoint's conversation
template and image tokens.

After preparing the documented source-build environment, build the native
targets `trtmc`, `trtmc_backend_trt`, and `trtmc_model_llava`, then run:

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --precision fp16 --max-sequence-length 1024 --output llava.bundle

build/trtmc run llava.bundle --runtime-root build \
  --image families/llava/tests/data/test_img.jpeg \
  --prompt "What color is the car? Answer with one word." \
  --max-new-tokens 4 --temperature 0
```

The bundle contains `vision.plan`, `prefill.plan`, `engine.plan`, the native
tokenizer, and `runtime.json`. Prefill and decode use separate engine weights;
the FP16 bundle is approximately 25.7 GiB. Prompt tokens, the 576 image tokens,
and requested answer tokens must fit the selected context length. The maximum
checkpoint context is 4,096 tokens.
Cache exhaustion raises an error at the selected context limit. Native cache
unit tests cover zero initialization, allocation failures, and exhaustion:

```bash
cmake --build build --target test_llava_kv_cache
ctest --test-dir build -R '^test_llava_kv_cache_' --output-on-failure
```

For the official E2E, provide `TRTMC_LLAVA_MODEL_DIR`, `TRTMC_BINARY`, and
`TRTMC_RUNTIME_ROOT`. Set `TRTMC_REFERENCE_PYTHON` to an independent CUDA
PyTorch interpreter with Transformers 5.2.0, Pillow, and sentencepiece:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  families/llava/tests --e2e-model llava -q
```

Four image cases require exact greedy token equality with Hugging Face and
the expected semantic answer: white car, outdoors, red, and blue. The same
four cases run on the FP8 bundle (`manifests/llava-1.5-7b-fp8.json`): the
answer must contain the expected word, and the Hugging Face FP16 model, forced
through the native tokens, must rank every native token within 2.0 logit of
its top logit (`thresholds/llava-1.5-7b-fp8.json`; about twice the largest gap
of the FP8 engine's greedy choice on 6176 held-out positions, 1.05 logit).
Two native
input checks separately compare portrait/landscape image processing and
tokenization with Hugging Face. This does not establish performance or
deployment qualification, or support for other LLaVA variants and precisions.

## Quantization

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --precision fp16 --quantization fp8 --max-sequence-length 1024 \
  --output llava-fp8.bundle
```

No FP8 or NVFP4 checkpoint of LLaVA 1.5 exists, so `--quantization fp8`
quantizes the original checkpoint while it builds the prefill and decode
engines (`quantization.py`); FP8 weights are never stored. The recipe follows
the dense-text and VLM guidelines of ModelOpt (language model only, static
per-tensor scales, max calibration):

| Part | Precision |
| --- | --- |
| q, k, v, o, gate, up and down projections of all 32 decoder layers | FP8 W8A8: weight scale `amax / 448` per projection, static input scale, FP16 Dequantize |
| Down projection of layer 1 | FP16: its calibrated input scale is 27.5 times the median of its kind (the limit is 20 times) |
| CLIP tower, projector, token embedding, head, norms, attention core | FP16 as in the unquantized build; the vision engine is 2.5 ms of a 350 to 450 ms request |

The input scales (`fp8_activation_scales.json`) are `amax / 448` from ModelOpt
`FP8_DEFAULT_CFG` on the FP16 model, over 384 COCO validation images, each with
a question in the checkpoint's conversation template and a teacher-forced
answer (news and Wikipedia text up to 700 tokens). Rerun `calibrate_scales.py`
for another checkpoint: an entry is keyed by the decoder config and also stores
the first 8 values of the final norm weight, so another checkpoint of the same
config is refused, as are a checkpoint that is already quantized and any
precision but `fp16`. The attention is explicit (FP32 scores), so the attention
output projection is quantized as well. FP8 builds turn the fused gate/up (dual
GEMM) kernel off: on TensorRT 11.4 the build without it decodes 19% faster on
B200 and as fast on B300.

Measured with TensorRT 11.4, `max_sequence_length` 1024, batch 1, the engines
run by a Python driver that feeds them like the runtime (split prefill and
decode engines, cache rows in buckets of 256; all variants of a GPU built and
timed in one job, medians of 7 interleaved rounds). A request is the vision
engine, a 596-row prompt and 64 decode steps:

| Engine | Prefill + decode plans | Decode step B200 / B300 | Prompt B200 / B300 | Request B200 / B300 |
| --- | --- | --- | --- | --- |
| FP16 | 25718 MiB | 5.651 / 5.130 ms | 16.29 / 15.17 ms | 445.6 / 411.2 ms |
| FP8 | 13461 MiB (-48%) | 4.235 / 4.240 ms (1.33x / 1.21x) | 12.60 / 12.17 ms (1.29x / 1.25x) | 352.2 / 351.4 ms (1.27x / 1.17x) |

The bundle shrinks from 27.6 GB to 14.7 GB (-47%). TensorRT 11.1 builds the FP8
engines too (B300: decode 1.16x, KL 0.0049, no non-finite value).

Quality against the original FP16 model in transformers (eager attention):
KL(reference || engine) per token and top-1 agreement on 5376 held-out
teacher-forced tokens (24 COCO images with questions and 224-token news or
Wikipedia answers; none used for calibration). Ranges are B200 to B300:

| Engine | KL mean | KL p99 | Top-1 |
| --- | --- | --- | --- |
| transformers FP16 vs FP32 | 0.00004 | 0.0003 | 0.996 |
| FP16 engine | 0.00005 to 0.00029 | 0.0004 to 0.0017 | 0.990 to 0.996 |
| FP8 (shipped) | 0.0052 to 0.0054 | 0.041 to 0.044 | 0.963 |
| FP8 without q, k, v | 0.0035 to 0.0038 | 0.028 to 0.029 | 0.967 to 0.969 |

Two 400-token answers (cache rows up to about 1000) give KL 0.0043 to 0.0045,
and 164 positions scored through the prefill engine 0.0053 to 0.0055. Greedy
captions of 14 other images equal the FP16 engine's caption in 11 to 12 cases.
No logit was non-finite, also for noise, black and white images decoded up to
the full 1024-row cache. Keeping q, k and v in FP16 lowers the KL by 30% but
also the decode gain (1.29x / 1.11x instead of 1.33x / 1.21x).

NVFP4 was measured and is not shipped: it needs a KL of at most about twice
that of FP8, and the best layout is 5.6 times. Its decode step is 0.98x to
1.08x that of FP8, since the FP16 decode step (5.1 to 5.7 ms) is far from the
time to read the weights. NVFP4 uses `add_dynamic_quantize`, per-16 E4M3 block
scales, `weight_scale_2 = amax / (6 * 448)` and the FP8 input scale over 6:

| NVFP4 layout | Plans | Decode step vs FP8, B200 / B300 | KL mean | Top-1 |
| --- | --- | --- | --- | --- |
| All seven projections NVFP4 | 8091 MiB | 1.08x / 1.00x | 0.0502 | 0.885 to 0.889 |
| Gate, up, down NVFP4; q, k, v, o FP8 | 9883 MiB | 1.04x / 1.00x | 0.0334 to 0.0337 | 0.908 to 0.910 |
| Gate, up, down NVFP4; q, k, v, o FP16 | 13977 MiB | 1.00x / 0.92x | 0.0311 | 0.908 |
| As the second, layers 0, 1, 30, 31 MLP in FP8 | 10297 MiB | 0.98x / 1.00x | 0.0300 to 0.0304 | 0.913 to 0.916 |

`tests/test_quantization.py` covers the FP8 recipe, the scale file, the
rejected requests and tiny FP16 / FP8 prefill and decode engines.
