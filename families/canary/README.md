# Canary

This family builds `nvidia/canary-1b-v2` (978M parameters, NeMo
FastConformer encoder-decoder speech recognition) for the `transcription` task
as two TensorRT engines: the encoder (32 conformer layers with 8x convolution
subsampling and relative positional attention) and the decoder (8 pre-LN
Transformer layers with self-attention KV cache and cross-attention to the
encoder output). The manifests build fp16, and fp32 with the tensor-parallel
decoder (4 GPUs).

## Quantization

`--precision fp16 --quantization fp8` quantizes the original `.nemo` checkpoint
at build time (#1854). Encoder q/k/v/o and the four feed-forward matrices of each
layer, and decoder self-attention q/k/v, cross-attention q/k/v and fc1/fc2 get
per-tensor FP8 weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, 192 LibriSpeech
validation clips). The decoder attention output projections, convolutions,
positional projection, embeddings, norms and output projection stay fp16.
B300, TensorRT 11.1 and 11.4: the bundle is 40% smaller (1.82 to 1.09 GiB), the
encoder time does not change and the decoder step is about 1.05x faster;
LibriSpeech test WER is 2.72% to 2.85% against 2.78% for fp16.

### NVFP4

NVFP4 was not tried. It was a non-goal of the FP8 change, for these reasons
(numbers from #1854, B300, batch 1):

| Item | fp16 | FP8 | FP8 vs fp16 |
|---|---|---|---|
| Encoder, one 30 s window, TensorRT 11.1 / 11.4 | 5.55 / 5.51 ms | 5.45 / 5.52 ms | 1.02x / 1.00x |
| Decoder step, TensorRT 11.1 / 11.4 | 0.795 / 0.786 ms | 0.745 / 0.747 ms | 1.07x / 1.05x |
| Bundle | 1.82 GiB | 1.09 to 1.10 GiB | -40% |

The encoder is not bound by its projection GEMMs: with FP8 weights its time
did not change, because the relative-position attention matmuls, the
convolution modules, the subsampling and the layer norms stay fp16. The decoder step gains only about 1.05x from
half-size weights, so a further halving with NVFP4 has little to save.
