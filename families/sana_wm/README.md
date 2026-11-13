# SANA-WM

This family builds `Efficient-Large-Model/SANA-WM_bidirectional` for
`world_model_generation`: an image, a prompt and a camera path give a 321-frame
704 x 1280 video. The bundle holds TensorRT plans for the stage-1 text encoder
(Gemma 2), the stage-1 DiT (20 blocks, 60 steps, batch 2 for CFG), the refiner
text encoder and text connector, the refiner DiT (48 LTX-2 blocks, 3 steps) and
the LTX-2 VAE; the native C++ runtime runs the pipeline. All plans are bf16. Most
of the DiT compute runs in family-owned TensorRT plugins (`native_plugins/`)
that call ATen (PyTorch C++) kernels. The family has no quantized build; the
build refuses `quantization`.

## FP8 quantization study (not shipped)

FP8 was studied for the stage-1 DiT and is not shipped, and no FP8 build was
made: about 80% of the DiT GEMM and convolution time runs inside the family
plugins, where TensorRT Quantize/Dequantize does not apply. Only `attn.qkv` and
the camera `q/k/v_proj_cam` projections are TensorRT graph matmuls. Making
those four FP8 saves at most about 1 ms per block (7% of the GEMM and
convolution time, a smaller share of the whole DiT call), and it would need a
new calibration path for a 36,080-token input. This section records the
evidence, so that the study does not have to be repeated.

### Summary and decision

Stage-1 DiT, one block, manifest config (36,080 tokens, batch 2 = 72,160 rows),
TensorRT 11.4:

| Item | B200 | B300 |
|---|---|---|
| Graph GEMMs (`attn.qkv`, `q/k/v_proj_cam`), bf16 | 2.98 ms | 3.22 ms |
| Same GEMMs in FP8 (TensorRT Quantize/Dequantize) | 1.93 ms | 2.14 ms |
| GEMMs and convolutions inside plugins (bf16 ATen) | 11.43 ms | 10.74 ms |
| Graph share of the GEMM and convolution time | 21% | 23% |
| FP8 saving per block, share of the GEMM and convolution time | 1.06 ms, 7.3% | 1.09 ms, 7.8% |
| FP8 saving per 60-step video (20 blocks x 60 calls) | at most 1.3 s | at most 1.3 s |

Decision: do not ship FP8 for SANA-WM. FP8 on the graph part cannot meet the
speed bar that the other FP8 builds meet (the DiT call also has the GDN scan,
attention, camera and normalization plugins, which FP8 does not touch), and a
quality study of a 321-frame video model would cost more than the gain. The
refiner DiT has no graph GEMM at all.

### Where the DiT compute runs

Per-token linear weights of one stage-1 block (from the checkpoint header,
`dit/sana_wm_1600m_720p.safetensors`) and where they run on `main`:

| Weights | Params (M) | Runs in | FP8 Q/DQ possible |
|---|---|---|---|
| `mlp.inverted_conv` (1x1, 2240 to 13440), `mlp.depth_conv` (3x3 depthwise), `mlp.point_conv` (1x1, 6720 to 2240), `mlp.t_conv` (3x1 temporal) | 30.1 + 0.1 + 15.1 + 15.1 | `SanaWmGlumbconvTemp` plugin: weights are plugin fields, `at::conv2d` in `enqueue` | no |
| `cross_attn.q_linear`, `cross_attn.proj` (and `kv_linear` on 300 text tokens) | 5.0 + 5.0 (+ 10.0) | `SanaWmCrossAttention` plugin: `at::matmul`, `at::linear`, `at::scaled_dot_product_attention` | no |
| `attn.output_gate`, `attn.proj`, `attn.out_proj_cam`, `plucker_proj` | 4 x 5.0 | `SanaWmGateProj` bf16 linear plugin: `at::linear` | no |
| `attn.qkv` | 15.1 | TensorRT graph matmul (`_add_linear`; the plugin path is built with `enabled=False`) | yes |
| `attn.q_proj_cam`, `k_proj_cam`, `v_proj_cam` | 3 x 5.0 | TensorRT graph matmul (`_add_linear`) | yes |

The graph path also has the GDN scan (`SanaWmGdnScan` and its combined
variants), softmax attention in every fourth block (`SanaWmSoftmaxAttention`),
UCPE camera transforms, layer norm, modulation and frame gate as plugins. The
small per-sample linears (timestep, caption and final layer) are graph matmuls,
but run on a few rows only.

The refiner DiT (37.8 GB, `refiner_dit_builder.py`) packs each of its 48 LTX-2
blocks into one `SanaWmLtxVideoBlock` plugin (`packed_weights` field; all
linears are `at::linear` and attention is `at::scaled_dot_product_attention`),
so it has no graph GEMM that could take FP8. The refiner text connector is also
built from plugins.

### Method

No FP8 or NVFP4 checkpoint of SANA-WM exists on the Hub (the
Efficient-Large-Model releases are bf16 or fp32), and no FP8 build of the family
was made. The full stage-1 DiT plan was not profiled; the GEMMs of one block were
timed in isolation:

- Shapes from the checkpoint header and `config.yaml` at the manifest config:
  321 frames, 704 x 1280, VAE stride 8 x 32 x 32, so 41 x 22 x 40 = 36,080 latent
  tokens per sample, batch 2 (CFG), hidden size 2240.
- Graph GEMMs: one strongly typed TensorRT network per GEMM shape, input
  `[72160, K]` bf16 times a constant `[K, N]` weight. FP8: input Quantize and
  Dequantize (per-tensor), FP8 weight constant with a per-tensor Dequantize to
  bf16, the form used by the other FP8 builds. Random weights; timing only, no
  quality measured.
- Plugin GEMMs and convolutions: the same ATen calls as the plugin `enqueue`
  code in bf16 (`at::conv2d` in channels-last for the GLUMBConvTemp FFN,
  `at::matmul`/`at::linear` for the 2240 x 2240 projections).
- B200 and B300, one GPU per job, TensorRT 11.4.0.106 (target) and 11.1.0.106,
  torch 2.14.0 (CUDA 13). CUDA events, median of 9 interleaved rounds of 10
  calls.

### Results

GEMM time per call, ms (x is bf16 time over FP8 time):

| GEMM (count per block) | B200 bf16 | B200 FP8 | x | B300 bf16 | B300 FP8 | x |
|---|---|---|---|---|---|---|
| `attn.qkv` 2240 to 6720 (1) | 1.419 | 0.835 | 1.70 | 1.570 | 0.938 | 1.67 |
| `q/k/v_proj_cam` 2240 to 2240 (3) | 0.521 | 0.364 | 1.43 | 0.551 | 0.399 | 1.38 |

Plugin GEMMs and convolutions per block, bf16 ATen, ms:

| Plugin op | B200 | B300 |
|---|---|---|
| FFN `inverted_conv` 1x1, 2240 to 13440 | 3.188 | 3.014 |
| FFN `depth_conv` 3x3 depthwise, 13440 channels | 1.891 | 1.832 |
| FFN `point_conv` 1x1, 6720 to 2240 | 1.602 | 1.536 |
| FFN `t_conv` 3x1, 2240 to 2240 | 1.593 | 1.410 |
| cross attention `q_linear` + `proj` | 1.067 | 0.926 |
| cross attention `kv_linear` (300 text tokens) | 0.017 | 0.015 |
| `output_gate`, `proj`, `out_proj_cam`, `plucker_proj` | 2.076 | 2.005 |
| sum | 11.43 | 10.74 |

TensorRT 11.1 gives the same picture: graph GEMMs 3.04 ms bf16 and 1.95 ms FP8
on B200 (3.13 and 2.13 ms on B300), plugin ops 11.48 ms (10.78 ms), FP8 saving
7.5% (7.2%) of the GEMM and convolution time.

### Why FP8 does not pay here

- TensorRT Quantize/Dequantize only reaches graph layers. The plugins hold their
  weights as bf16 plugin fields and call ATen kernels in `enqueue`, so 79% (B200)
  to 77% (B300) of the block GEMM and convolution time cannot become FP8 without
  changing the plugins.
- The two graph GEMMs that can be FP8 are 1.4x to 1.7x faster in isolation,
  which is about 1 ms per block. The whole DiT call has more non-GEMM work (GDN
  scan, attention, UCPE, norms), so the end-to-end gain is below 7%.
- A quality check would need the full 102 GB checkpoint, the 60-step CFG
  pipeline and a calibration set at the 36,080-token shape for this small gain.

### NVFP4 (not tried)

NVFP4 has the same limit: TensorRT block quantization applies only to graph
matmuls, which are the same two projections.

### What would change it

If the FFN, cross-attention and gate projections were graph matmuls (or the
plugins had FP8 GEMM paths), the same isolated timing shows that FP8 could save
about 4.7 ms of the 14.4 ms of block GEMM and convolution time on B200 (about
33%): `inverted_conv` 3.09 to 1.73 ms, `point_conv` 1.70 to 1.15 ms, `t_conv`
1.76 to 1.13 ms, the six 2240 x 2240 projections 0.58 to 0.39 ms each (TensorRT
11.4, as graph GEMMs). The depthwise convolution (1.9 ms) has no FP8 kernel. Such
a change would also need calibration and a video quality study (the
`minimax_h3` FP8 build is an example).

Revisit FP8 when:

- the GLUMBConvTemp FFN, cross attention or `SanaWmGateProj` linears are lowered
  as TensorRT graph layers, or
- the family plugins gain FP8 GEMM paths.
