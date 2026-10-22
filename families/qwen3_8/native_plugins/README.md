# Qwen3.8 Gated DeltaNet plugins (cuTile)

Opt-in TensorRT plugins for the Gated DeltaNet (GDN) block of Qwen3.8. Both replace steps 3-10 of
the block (conv1d + SiLU, gate/beta, delta rule, gated RMSNorm). The input projections (qkv, z, a,
b) and `out_proj` stay TensorRT matmuls. Batch is 1.

| Plugin | Tokens per call | Python entry point | Kernels |
|---|---|---|---|
| `Qwen38GdnDecodeTileGym` | 1 | `add_gdn_decode_plugin` | `cutile/gdn_tilegym.py` (4) |
| `Qwen38GdnPrefillTileGym` | fixed T, or a runtime T up to the engine maximum (`dynamic`); `valid_len` for a padded last chunk | `add_gdn_prefill_plugin` | `cutile/gdn_prefill.py` (6: conv, conv state, gate, intra-chunk, inter-chunk, norm) |

The kernels are adapted from [NVIDIA/TileGym](https://github.com/NVIDIA/TileGym) (MIT).

## Files

| File | Role |
|---|---|
| `../cutile/gdn_tilegym.py`, `../cutile/gdn_prefill.py` | cuTile kernels (decode, prefill) |
| `gdn_tilegym_plugin.cpp`, `gdn_prefill_plugin.cpp` | `IPluginV3` plugins: load the cubins with the CUDA driver API and launch them in order |
| `cubin_launch.h` | Shared cubin loading and launch helpers |
| `gdn_plugin_builder.py` | Build time: compiles the cubins for the device arch, builds and loads the plugin `.so`, adds the layer |
| `CMakeLists.txt` | Builds the plugin `.so` (needs TensorRT and CUDA headers, links libcuda) |
| `../tests/test_gdn_plugins.py` | One-plugin engines vs an fp32 reference (decode with and without the fused a/b projection, prefill padding cases, chunked prompt) |
| `../tests/test_gdn_projection_merge.py` | One FP8 GEMM for two projections (per-row weight scales) vs two GEMMs |

The cubins are stored in the serialized plugin fields, so the engine is self-contained. The
runtime process must load the plugin library before it deserializes the engine. The C++ runtime
does that itself: `runtime/CMakeLists.txt` builds `libtrtmc_qwen3_8_native_plugin.so` next to the
family library (option `TRTMC_QWEN3_8_NATIVE_PLUGIN`, on by default; skipped without libcuda or
the TensorRT and CUDA headers), and `plugin_helpers.cpp` loads it before the first engine. Engines
without plugins do not need it.

The prefill plugin is what makes the prefill engine dynamic: by default (`--prefill-engine auto`)
the build makes an engine that takes any token count from 1 to `--prefill-chunk-len` (default
1024) in one call, and the plugin reads the count at run time. A delta rule built from separate
TensorRT ops has a fixed number of blocks per graph, so it cannot follow a runtime count. The
build uses the static prefill engine instead, and says why, when the plugin cannot be used (fp32,
MTP engines, no cuda-tile package or plugin build tools); `--prefill-engine dynamic` fails in
those cases and `--prefill-engine static` skips the plugin. `TRTMC_QWEN3_8_GDN_CUTILE_PREFILL=1`
only adds the plugin to a static prefill engine.

## Use

| Environment variable | Effect |
|---|---|
| `TRTMC_QWEN3_8_GDN_CUTILE=1` | Use the decode plugin for the 48 GDN layers (fp16 or bf16 activations only). Default off: the original separate TensorRT ops. |
| `TRTMC_QWEN3_8_GDN_CUTILE_PREFILL=1` | Use the prefill plugin in a static prefill engine (fp16 or bf16 only). The dynamic prefill engine always uses it. Default off. |
| `TRTMC_QWEN3_8_GDN_MERGE_PROJ=1` | With the decode plugin, fewer input GEMMs. No quantization: one GEMM for qkv, z, a, b, then slice. FP8 weights (qkv and z with the same input scale): one FP8 GEMM for qkv and z (rows concatenated, one weight scale per row) and the plugin computes a and b itself. Otherwise the four projections stay separate. |
| `TRTMC_QWEN3_8_GDN_CONV_FP32=1` | Keep the conv weight and bias as fp32 checkpoint values. Default (bf16 models): the host rounds them to bf16, as the original ops store them. The kernel still accumulates in fp32. |
| `TRT_ROOT`, `TRT_LIB_DIR`, `CUDA_HOME` | Where to find TensorRT headers, `libnvinfer.so.11` and `cuda.h` if they are not in the usual places |
| `TRTMC_GDN_CONV_BLOCK`, `TRTMC_GDN_BLOCK_V`, `TRTMC_GDN_STREAM_V` | Decode kernel tile sizes (defaults 256, 32, 16) |
| `TRTMC_GDN_PREFILL_BLOCK_V` | Prefill delta-rule V tile (default 32) |
| `TRTMC_GDN_PREFILL_CHUNK` | Prefill delta-rule chunk length in tokens (default 32) |

Decode interface: 11 inputs (`qkv, z, a, b, conv_state, conv_w, conv_b, A, dt_bias, gamma,
ssm_state`), 3 outputs (`out`, `conv_state'`, `ssm_state'`). With `hidden_size` set, inputs 2 and 3
are the normalized hidden row `x[1, hidden]` and the fp32 weight `w_ab[2H, hidden]` (a rows, then
b rows) instead of `a` and `b`; the gate kernel then computes one dot product per head and rounds
a and b to the activation dtype, as a GEMM output would be.
Prefill interface: the same plus a 12th input `valid_len` (int32, 1 <= valid_len <= T); `qkv`,
`z`, `a`, `b` and `out` have T rows.
Activations and the conv state use the activation dtype; weights and the recurrent state are fp32.

## Prefill chunks and padding

A prompt of N tokens runs as ceil(N / T) calls, carrying `conv_state` and `ssm_state` from one
call to the next. (A dynamic engine takes up to its maximum T in one call; the runtime feeds longer
prompts in chunks of that maximum plus a shorter last chunk.) In the last call `valid_len` is the number of real tokens: the delta rule runs
only over the real tokens, the new conv state is the last four real inputs, and output rows at
or after `valid_len` are zero. The cost scales with `valid_len`.

## Notes on the cubins

- Every kernel needs the block size the tile compiler chose (128 or 256 threads). The builder
  reads it from each cubin and passes it to the plugin.
- Array arguments are declared with 16-byte alignment and stride divisibility
  (`base_addr_divisible_by`, `stride_divisible_by`). Without these hints the compiler cannot
  vectorize the accesses, and the cubins ran about 3x slower than `ct.launch` (prefill delta
  rule) and lost most of the decode speedup.

## Measured results (TensorRT 11.6, bf16, batch 1, CUDA graph)

Decode, GDN core only (no GEMMs, MLP or layer norms), us:

| GPU | Original ops | Plugin |
|---|---|---|
| B100 | 24.6 | 16.1 |
| RTX 5090 | 16.4 | 9.1 |

Decode, one GDN decoder layer (original ops = 1.00x, above 1.00x is faster). Timings on a shared
GPU move by about 10% between runs, so ranges of two runs are given:

| GPU | Layout | Plugin | Plugin + `MERGE_PROJ=1` |
|---|---|---|---|
| B100 | bf16 | 0.90-1.00x | 1.04-1.15x |
| B100 | FP8 | 0.98-1.13x | 1.08-1.35x |
| B100 | NVFP4 (FP8 qkv, z, out_proj) | 1.00-1.25x | 1.30-1.49x |
| RTX 5090 | bf16 | 0.99x | 1.03x |
| RTX 5090 | FP8 | 1.00x | 1.19x |
| RTX 5090 | NVFP4 (FP8 qkv, z, out_proj) | 1.03x | 1.02x |

The plugin boundary stops TensorRT from fusing the four input GEMMs, so the plugin alone is about
even with the original ops. `MERGE_PROJ=1` fixes that:

- bf16: one GEMM for qkv, z, a, b. TensorRT splits a two-projection GEMM (qkv, z) into two slow
  kernels, so a and b stay inside the merged GEMM.
- FP8 and NVFP4 layouts: one FP8 GEMM for qkv and z (removes one GEMM), and the plugin computes a
  and b (removes the a and b GEMM, which was about 24 us). The a and b dot product costs 1.3 us
  more than the gate kernel alone.

On the RTX 5090 a layer is bound by weight reads (about 770 MB of bf16 weights per layer, about
430 us at 1.8 TB/s, against about 500 us measured), so the 7 us saved in the GDN core is at most
a few percent. In the NVFP4 layout TensorRT already fuses the qkv and z GEMMs in the original
path, so the merge adds nothing there.

Without a CUDA graph the decode plugin adds CPU launch time (four launches per layer).
The hidden-state difference to the original ops is the same with and without the merge for the
bf16 and FP8 layouts (bf16 rounding). In the NVFP4 layout the merged variant differs by 0.29 at
an absmax of 6.75 (1.6e-2 without the merge). A one-step rounding change in a or b can move an
element across an FP4 quantization boundary in the MLP input; this has not been checked on the
full model.

Prefill, GDN core only, T = 128, us (first version, one sequential delta-rule kernel; the chunked
kernel now in `cutile/gdn_prefill.py` takes about 45 us per layer at T = 128 on an RTX 5090). The baseline is the T single-token steps chained in one
network, as the multi-token engine builds them:

| GPU | Original ops x 128 | Plugin, valid_len = 128 | valid_len = 64 | valid_len = 1 |
|---|---|---|---|---|
| B100 | 1649 | 254 | 100 | 20 |
| RTX 5090 | 1307 | 152 | 86 | 20 |

Accuracy against an fp32 token-by-token reference: out error about 0.5% of the output range in
bf16 (0.07% in fp16), state error about 1e-2 (bf16) and 1e-3 (fp16), conv state exact, at every
`valid_len` and for a chunked 300-token prompt.

The FP8 merge needs equal input scales for qkv and z; otherwise the projections stay separate.

Not covered: a full-model comparison and the B200 numbers after the alignment fix.
