# SeamlessM4T v2

Text-to-text translation with
[facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large):
the 24-layer text encoder and the 24-layer multilingual text decoder only (no
speech encoder, speech generation or vocoder). `model.py` builds two FP32
TensorRT engines: the encoder runs once per request, and the decoder recomputes
all `max_sequence_length` rows (3 to 4096, default 128) at every step (no KV
cache) and returns the logits of the last row. The runtime
(`runtime/plugin.cpp`) tokenizes with the checkpoint's BPE, adds the source
language token, and decodes greedily after the target language token. Only
`precision=fp32`, batch 1 and one GPU are supported. The FP32 bundle
reproduces the Hugging Face FP32 greedy tokens exactly (`tests/test_e2e.py`).

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  facebook/seamless-m4t-v2-large --revision 5f8cc790b19fc3f67a61c105133b20b34e3dcb76 \
  --task text_translation --precision fp32 --max-sequence-length 128 \
  --output seamless-m4t-v2-large-text.bundle
```

## Quantization

`--quantization fp8` builds the same two engines with FP8 projections; the
default stays unquantized (FP32), and the unquantized networks are unchanged.

```bash
PYTHONPATH=core/builder:. python3 -m tensorrt_model_connect build \
  facebook/seamless-m4t-v2-large --revision 5f8cc790b19fc3f67a61c105133b20b34e3dcb76 \
  --task text_translation --precision fp32 --quantization fp8 \
  --max-sequence-length 128 --output seamless-m4t-v2-large-text-fp8.bundle
```

No FP8 or NVFP4 checkpoint of SeamlessM4T v2 exists, so `--quantization fp8`
quantizes the original FP32 checkpoint while it builds the engines
(`quantization.py`); FP8 weights are never stored. The recipe follows the
ModelOpt guidelines for encoder-decoder audio and translation models (all
matmul projections, static per-tensor scales, max calibration):

| Part | Precision |
| --- | --- |
| Encoder self-attention q, k, v, out and ffn fc1, fc2 (24 layers); decoder self-attention q, k, v, out, cross-attention q, k, v, out and ffn fc1, fc2 (24 layers): 384 projections | FP8 W8A8: weight scale `amax / 448`, static input scale, FP32 Dequantize |
| q, k and v of one self-attention; k and v of one cross-attention | One shared weight scale (the largest amax of the group), as they read one input |
| Token embedding, sinusoidal positions, head, layer norms, attention core | FP32 as in the unquantized build |

The input scales (`fp8_activation_scales.json`) are `amax / 448` from ModelOpt
`FP8_DEFAULT_CFG` on the FP32 model in PyTorch, over 512 WMT16 validation
sentences (cs, de, fi, ro, ru, tr to and from English) translated greedily, so
the decoder and the cross-attention see decode-time inputs
(`calibrate_fp8_scales.py`). No projection is an outlier (the largest input
scale is 6.1 times the median of its kind; the limit is 20 times). An entry is
keyed by the model config and also stores the first 8 values of the final
decoder norm weight, so another checkpoint of the same config is refused, as
are a checkpoint that is already quantized and any format but `fp8`. The
attention is explicit (FP32 scores), so the attention output projections are
quantized as well.

Measured with TensorRT 11.4, batch 1, the engines driven like the runtime
(source encoded once, decoder inputs copied in and logits copied to the host
every step, argmax on the host; all builds of a GPU in one job, medians of 7
interleaved rounds). The request is the first premerge input (English to
French) with 64 decoder steps; the real translation stops after 16:

| Build | Plans | Bundle | Request (64 steps) B200 / B300 | Request (16 steps, to EOS) B200 / B300 |
| --- | --- | --- | --- | --- |
| FP32 (default) | 6233 MiB | 6.55 GB | 482.0 / 462.0 ms | 124.6 / 119.4 ms |
| FP8 | 3075 MiB | 3.24 GB (-51%) | 133.6 / 125.8 ms (3.61x / 3.67x) | 34.3 / 32.3 ms (3.63x / 3.70x) |

One decoder step takes 7.33 / 7.03 ms in FP32 and 1.95 / 1.83 ms in FP8, the
encoder 5.40 / 5.19 ms and 1.09 / 1.03 ms. Quantizing only the decoder gives
3.5x; the encoder adds the rest. TensorRT 11.1 builds the FP8 engines too
(B300: request 129.1 vs 463.5 ms, 3.59x; KL 0.0035 and no non-finite value
on the set below).

Quality on 224 held-out WMT14 and WMT16 test sentences (16 per direction, 14
directions: cs, de, fi, fr, ro, ru, tr to and from English), none used for
calibration. KL is KL(reference || engine) per token on the 6302 tokens of the
PyTorch FP32 greedy translations, teacher-forced; the original checkpoint is
FP32, so PyTorch FP32 (eager attention, no TF32) is the reference. BLEU and
chrF are against the WMT references (sacrebleu). B200 and B300 give the same
values:

| Engine | KL mean | KL p99 | Top-1 | Same greedy translation as PyTorch FP32 | BLEU | chrF |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0 | 0 | 1.000 | 100% | 32.83 | 58.28 |
| PyTorch BF16 | 0.00013 | 0.0008 | 0.998 | 92.4% | 32.56 | 58.11 |
| FP32 engine (default) | 0.00000 | 0.00000 | 1.000 | 100% | 32.83 | 58.28 |
| FP8 engine (shipped) | 0.0036 | 0.025 | 0.989 | 73.2% | 32.83 | 58.45 |

The translation quality against the WMT references does not change. No logit
or encoder output was non-finite in the 6302 teacher-forced steps, in the 224
greedy translations, or in 9 stress requests decoded for the full 126 steps
(random token ids, one repeated token and 125 tokens of repeated text, each to
French, Chinese and Russian). The FP8 translations of the three premerge
inputs equal the FP32 ones.

Also measured, not used: a weight scale per q, k and v projection (KL 0.0036,
same speed; the shared scale is kept, since per-projection q, k and v scales
gave a wrong engine in the qwen2_5_omni family), and FP16 instead of FP32 Dequantize (same KL, 1.13x to
1.15x slower per request).

NVFP4 was measured and is not shipped: it is not faster than FP8 and loses 4.5
to 7.8 times the FP8 KL (the limit is about 2 times). In this FP32 network
TensorRT 11.4 finds no kernel for the NVFP4 dynamic quantize of an FP32
activation (`Could not find any implementation`), so the NVFP4 projections
cast their input to FP16 first. NVFP4 uses `add_dynamic_quantize`, per-16
E4M3 block scales, `weight_scale_2 = amax / (6 * 448)` and the FP8 input scale
over 6; the projections not listed as NVFP4 stay FP8. Request ratios are
against the FP8 build of the same job:

| NVFP4 layout | Plans | Request vs FP8, B200 / B300 | KL vs FP32 | KL / FP8 KL | Top-1 | BLEU |
| --- | --- | --- | --- | --- | --- | --- |
| All projections | 2618 MiB | 0.86x / 0.80x | 0.0281 | 7.8x | 0.959 | 32.17 |
| All projections, BF16 input | 2619 MiB | 0.84x / 0.80x | 0.0273 | 7.6x | 0.961 | 31.72 |
| fc1 and fc2 | 2740 MiB | 0.99x / 0.92x | 0.0208 | 5.8x | 0.965 | 32.19 |
| fc1 and fc2, layers 2 to 21 | 2795 MiB | 0.99x / 0.93x | 0.0160 | 4.5x | 0.970 | 32.59 |
| fc1 only | 2906 MiB | 1.01x / 0.96x | 0.0174 | 4.9x | 0.970 | 32.43 |

### FP16 (finding, not the default)

An FP16 build of the same networks (FP16 weights and activations, FP32 layer
norm statistics, FP32 engine inputs and outputs; study code, not in this
family) is faster than FP8 and almost lossless: request 93.0 / 87.5 ms on
B200 / B300 (5.2x / 5.3x faster than FP32, 1.44x / 1.44x faster than FP8),
plans 3121 to 3621 MiB, KL vs FP32 0.00001 to 0.00003, top-1 0.999, BLEU 32.87 to
32.96, no non-finite value in the same scans. A BF16 build is 4.7x to 4.8x
faster than FP32 at KL 0.00013. FP8 is slower than FP16 here because the FP8
GEMMs write FP32 and the rest of the FP32 network (attention, norms, head)
stays FP32. Making FP16 the default is a separate change.

`tests/test_quantization.py` covers the FP8 recipe, the scale file, the shared
weight scales, the rejected requests and tiny FP32 / FP8 engines; the
`seamless-m4t-v2-large-text-fp8` manifest runs the FP8 bundle through the
native runtime (the first 6 tokens equal to PyTorch FP32, the end-of-sequence
token reached and the expected word present).
