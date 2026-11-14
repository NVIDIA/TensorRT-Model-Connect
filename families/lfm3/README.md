# LFM3 text generation

This family implements the native TensorRT graph and runtime for
`LiquidAI/LFM3-1.7B`, pinned in the E2E manifest to revision
`ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`. Actual TensorRT checkpoint parity
and GPU qualification are still required before treating this family as validated.

The initial scope is one request on one GPU, FP16 or BF16, with a fixed cache.
Prefill processes each token in order to update convolution and attention state
together. Sliding attention masks the full allocated KV cache; it does not reduce
the cache allocation. Raw prompts and single-user ChatML prompts support the
checkpoint's thinking toggle. Nonempty system prompts, quantized checkpoints,
tensor parallelism, dynamic caches, batching and speculative decoding are rejected.

Use the public Python `BuildRequest` with `family="lfm3"`,
`task="text_generation"`, and `precision="bf16"` or `"fp16"`. Build and runtime
interfaces follow the repository's public Task API. The native family DSO is
`libtrtmc_model_lfm3.so`. Rebuild both the bundle and this DSO when changing
family topology or runtime metadata.

The E2E manifest declares raw, chat and sliding-window boundary cases with exact
greedy token-ID comparison to the pinned official reference. CPU tests exercise
configuration, safetensors mapping and graph helpers; they do not prove device
execution or checkpoint parity. E2E requires `TRTMC_NATIVE_BUILD_DIR` and runs
the native tokenizer checkpoint executable against the downloaded revision's
`tokenizer.json` before building an engine. All 36 fixtures must pass; missing
assets or a skipped executable fail E2E. For standalone execution, set
`TRTMC_LFM3_TOKENIZER_JSON` to that tokenizer asset.

Repository source retains its Apache-2.0 notices. Model weights are distributed
by LiquidAI under the LFM Open License v1.0; users must obtain them separately
under that license. No model weights or external implementation files are included.

The native regex categories use Unicode 16.0 L* and N* data from
[UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt),
redistributed under Unicode License v3 in `UNICODE_LICENSE.txt`. Its SHA-256 is
`ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`.
Regenerate with the command below, then apply clang-format:

```bash
python families/lfm3/tools/generate_unicode_data.py \
  UnicodeData.txt families/lfm3/runtime/unicode_categories.h
```

The categories
match the pinned tokenizer regex for every Unicode scalar. The notice is installed
with this family.

## Quantization

`quantization=fp8` (fp16 or bf16) quantizes `LiquidAI/LFM3-1.7B` at build time:
q/k/v, the SwiGLU `w1`/`w3`/`w2` and the conv-block in and out projections run
as FP8 GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, Dequantize in the activation type). `layer.22.w2`
(outlier), the attention output and gate projections, the depthwise
convolution, embedding, norms and `lm_head` stay 16-bit. On TensorRT 11.4,
batch 1, decode is 1.30x (B200) and 1.10x (B300) faster than fp16, the bundle
is 35% to 40% smaller and WikiText perplexity rises by 1.1%. Other checkpoints
raise and name the calibration tool.

### NVFP4

NVFP4 was measured on top of this FP8 recipe and is not shipped (no NVFP4 code
is in the family). The study built real engines with the MLP (`w1`, `w3`, `w2`)
in NVFP4 (FP8 scale per block of 16, FP4 activations quantized at run time with
the FP8 activation scales) and the other projections as in the FP8 build.
TensorRT 11.4, fp16, `max_sequence_length` 512; quality and the NaN scan on
B200.

| Variant | Plan GiB | Decode ms B200 (vs fp16) | Decode ms B300 (vs fp16) | WikiText KL (x FP8) / ppl | Held-out KL | ARC-Easy (of 300) | Prompts with non-finite values |
|---|---|---|---|---|---|---|---|
| fp16 | 3.22 | 1.790 | 1.410 | 0.0007 / 50.49 | 0.0004 | 115 | 0 of 180 |
| FP8 (shipped) | 1.92 | 1.382 (1.30x) | 1.283 (1.10x) | 0.0205 (1.0x) / 51.03 | 0.0088 | 112 | 0 of 180 |
| NVFP4 MLP, rest FP8 | 1.46 | 1.362 (1.31x) | 1.272 (1.11x) | 0.1416 (6.9x) / 54.58 | 0.0571 | 116 | 0 of 180 |

At the default `max_sequence_length` 4096 NVFP4 is slower than FP8: 1.462
against 1.451 ms (B200) and 1.404 against 1.373 ms (B300).

Reasons:

- Speed: NVFP4 is 24% smaller than FP8 but not faster: at most 1.5% faster at
  512 tokens and slower at 4096.
- Quality: WikiText KL is 6.9x the FP8 KL; against the Hugging Face model,
  perplexity rises by 8% (FP8: 1.1%) and top-1 agreement is 0.788 (FP8: 0.906).

The model owner publishes an NVFP4 checkpoint, `LiquidAI/LFM3-1.7B-NVFP4`
(ModelOpt export, 1.8 GB). It was not measured here, and this family rejects
quantized checkpoints; the self-quantized study above found no speed gain over
FP8.
