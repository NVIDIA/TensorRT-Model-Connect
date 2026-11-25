# Qwen2.5 Omni

Text continuation with the Thinker of
[Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B): the
28-layer text decoder only (no audio, image or video input, no Talker or
speech output). `model.py` builds one FP32 TensorRT engine that takes one token
per step and keeps a fixed KV cache of `max_sequence_length` rows (2 to 1024,
default 128); the prompt is fed token by token through the same engine. The
runtime (`runtime/plugin.cpp`) tokenizes with the checkpoint's byte-level BPE
and decodes greedily. Only `precision=fp32`, batch 1 and one GPU are
supported. The FP32 engine reproduces the Hugging Face FP32 greedy tokens
exactly (`tests/test_e2e.py`).

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  Qwen/Qwen2.5-Omni-7B --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --task text_continuation --precision fp32 --max-sequence-length 128 \
  --output qwen2.5-omni-7b-text.bundle
```

## Quantization

`--quantization fp8` builds the same engine with FP8 decoder projections; the
default stays unquantized (FP32).

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  Qwen/Qwen2.5-Omni-7B --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --task text_continuation --precision fp32 --quantization fp8 \
  --max-sequence-length 128 --output qwen2.5-omni-7b-text-fp8.bundle
```

No FP8 or NVFP4 checkpoint of Qwen2.5-Omni-7B exists (the published quantized
ones are AWQ, GPTQ and GGUF), so `--quantization fp8` quantizes the original
checkpoint while it builds the engine (`quantization.py`); FP8 weights are
never stored. The recipe follows the VLM / omni guidelines of ModelOpt
(language model only, static per-tensor scales, max calibration):

| Part | Precision |
| --- | --- |
| q, k, v, o, gate, up and down projections of all 28 Thinker layers | FP8 W8A8: weight scale `amax / 448`, static input scale, FP32 Dequantize |
| q, k and v of one layer | One shared weight scale (the largest amax of the three): with a scale per projection the engine gives wrong logits on TensorRT 11.1 and 11.4 (KL 8.6, see below) |
| Down projection of layer 26 | FP32: its calibrated input scale is 56 times the median of its kind (the limit is 20 times) |
| Token embedding, head, norms, attention core, KV cache | FP32 as in the unquantized build |

The input scales (`fp8_activation_scales.json`) are `amax / 448` from ModelOpt
`FP8_DEFAULT_CFG` on the BF16 Thinker in PyTorch, over 512 cnn_dailymail
training articles with their highlights in the chat format (448 of up to 512
tokens and 64 of up to 1024 tokens). Rerun `calibrate_scales.py` for another
checkpoint: an entry is keyed by the decoder config and also stores the first 8
values of the final norm weight, so another checkpoint of the same config is
refused, as are a checkpoint that is already quantized and any format but
`fp8`. The attention is explicit (FP32 scores), so the attention output
projection is quantized as well. FP8 builds turn the fused gate/up (dual GEMM)
kernel off, as the other FP8 families do.

Measured with TensorRT 11.4, batch 1, the engine driven like the runtime (one
token per step, present K/V copied to the cache, logits copied to the host and
the argmax taken there; all variants of a GPU built and timed in one job,
medians of 7 interleaved rounds). A request is the 16-token chat prompt
"What is 2 + 2?" and 64 new tokens (79 steps):

| `max_sequence_length` | Engine | Plan | Step B200 / B300 | Request B200 / B300 |
| --- | --- | --- | --- | --- |
| 1024 | FP32 | 29054 MiB | 7.862 / 7.635 ms | 621.3 / 603.1 ms |
| 1024 | FP8 | 10588 MiB (-64%) | 4.525 / 4.303 ms (1.74x / 1.77x) | 357.5 / 340.0 ms (1.74x / 1.77x) |
| 128 | FP32 | 29053 MiB | 6.754 / 6.695 ms | 533.7 / 528.9 ms |
| 128 | FP8 | 10588 MiB | 3.655 / 3.570 ms (1.85x / 1.88x) | 288.9 / 282.2 ms (1.85x / 1.87x) |

The bundle (`max_sequence_length` 128) shrinks from 30.5 GB to 11.1 GB (-64%).
TensorRT 11.1 builds the FP8 engine too (B300, `max_sequence_length` 1024: step
1.77x, KL vs FP32 0.0070 on 3069 tokens, no non-finite value). A control that
only stores the same projections as BF16 (BF16 GEMM, the rest FP32) is 1.52x to
1.58x faster than FP32, so FP8 is 1.14x to 1.19x faster than that.

Quality on 5409 held-out teacher-forced tokens (4 windows of 1024 tokens of
WikiText-2 test and 2 cnn_dailymail test articles in the chat format, none used
for calibration), engine with `max_sequence_length` 1024. KL is
KL(reference || engine) per token. The original BF16 model in transformers
(eager attention) is itself far from the FP32 one on this model (KL(FP32 ||
BF16) 0.188, top-1 0.859), and the FP32 engine matches transformers FP32
exactly, so both references are shown. Ranges are B200 to B300:

| Engine | KL vs BF16 | Top-1 vs BF16 | KL vs FP32 | KL p99 vs FP32 | Top-1 vs FP32 |
| --- | --- | --- | --- | --- | --- |
| FP32 (default) | 0.2643 | 0.859 | 0.00000 | 0.0000 | 1.000 |
| FP8 (shipped) | 0.2688 to 0.2691 | 0.852 | 0.0078 to 0.0079 | 0.058 | 0.951 |
| FP8, separate q/k/v weight scales | 8.6 | 0.003 | 8.8 | 15.9 | 0.003 |

A PyTorch fake-quant simulation of the same FP8 recipe on the FP32 model gives
KL 0.0074 vs FP32 (top-1 0.962), so the engine matches the expected FP8 loss.
No logit was non-finite in the 5409 steps, in 3 stress sequences of 1024
tokens (random ids, one repeated token, repeated text) and in greedy decoding
of 4 prompts (one up to the full 1024-row cache). FP8 greedy answers to the
two premerge prompts equal the FP32 ones ("2 + 2 equals 4.", "The capital of
France is Paris.").

NVFP4 was measured and is not shipped: it needs a KL of at most about twice
that of FP8, and the best layout is 4.7 times (against FP32; 5 times the
FP8 loss against BF16). It is only 1.01x to 1.06x faster than FP8 per step (1.07x to 1.10x at
`max_sequence_length` 128 on B200),
since the FP32 head (2.2 GB) and the FP32 attention stay. NVFP4 uses
`add_dynamic_quantize`, per-16 E4M3 block scales, `weight_scale_2 = amax / (6 * 448)`
and the FP8 input scale over 6 (`max_sequence_length` 1024, ranges B200 to B300):

| NVFP4 layout | Plan | Step vs FP8, B200 / B300 | KL vs FP32 | Top-1 vs FP32 | KL vs BF16 |
| --- | --- | --- | --- | --- | --- |
| All seven projections NVFP4 | 7888 MiB | 1.06x / 1.01x | 0.0753 | 0.862 to 0.864 | 0.313 to 0.314 |
| Gate, up, down NVFP4; q, k, v, o FP8 | 8235 MiB | 1.04x / 1.03x | 0.0587 to 0.0591 | 0.878 | 0.304 to 0.306 |
| As the second, layers 0, 1, 26, 27 MLP in FP8 | 8547 MiB | 1.01x / 1.02x | 0.041 to 0.045 | 0.900 to 0.901 | 0.284 to 0.290 |
| Gate, up NVFP4; the rest as in FP8 | 8999 MiB | 1.05x / 1.03x | 0.0368 to 0.0375 | 0.902 to 0.907 | 0.287 to 0.289 |

`tests/test_quantization.py` covers the FP8 recipe, the scale file, the shared
q/k/v weight scale, the rejected requests and a tiny FP32 / FP8 engine; the
`qwen2.5-omni-7b-text-fp8` manifest runs the FP8 bundle through the native
runtime.
