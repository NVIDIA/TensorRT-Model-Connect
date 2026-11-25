# Chroma

`ChromaPipeline` checkpoints (`lodestones/Chroma1-HD`): T5-XXL text encoder, the 8.9 B
`ChromaTransformer2DModel` MMDiT and the FLUX.1 VAE, served as the `text_to_image` Task.

## Bundle

`trtmc build <checkpoint> -o chroma.bundle [--precision bf16|fp32] [--image-height H --image-width W]
[--max-sequence-length L] [--quantization none|fp8|nvfp4]` builds three TensorRT plans from the checkpoint, the tokenizer and
`runtime.json`. BF16 (as stored) is the default; FP32 is the parity build. The image size (multiples of
16 up to 4096, default 1024 x 1024) and the text length (at most 512, default 512) are fixed per bundle.

## Runtime Config

| Field | Default | Meaning |
| --- | --- | --- |
| `negative_prompt` | `""` | classifier-free guidance prompt, used when `guidance_scale > 1` |
| `num_steps` | 35 | flow-matching Euler steps (1 to 1000) |
| `guidance_scale` | 5.0 | classifier-free guidance scale |
| `seed` | 0 | noise seed; unused when initial latents are given |
| `height`, `width` | bundle size | fixed by the bundle |

The defaults are those of `ChromaPipeline.__call__`; the model card recommends 40 steps and guidance 3.0.
`initial_latents` replays noise in the packed `[(H / 16) * (W / 16), 64]` float32 layout, the layout of the
`latents` argument of `ChromaPipeline`. Without it the family draws its own seeded Gaussian noise
(`std::mt19937` with `std::normal_distribution`: reproducible for one C++ standard library, not equal to a
torch generator); replay latents for images that must match another implementation or platform.

## Reference semantics kept

The reference is diffusers `ChromaPipeline` with transformers' `T5EncoderModel` and T5 tokenizer.

- The text attention mask covers the prompt tokens, `</s>` and the first padding token. The pipeline passes it
  to scaled-dot-product attention as a floating-point 0/1 tensor, so it acts as the additive term
  `m[i] * m[j]` on the scores; the family does the same.
- The T5 feed-forward output projections run in FP32 in a BF16 build, as transformers keeps them FP32.
- `torch.nn.RMSNorm` layers (query/key norms, the approximator norms) apply their weight in FP32 and round
  once, as on CUDA; the approximator norms use the float32 epsilon (`eps=None`), also for BF16 inputs.
- A BF16 bundle keeps the sampler state in BF16 where the pipeline does: the initial latents, the guided
  prediction, the Euler update, the VAE input scaling, the image denormalization, and the timestep before
  its sinusoid. An FP32 bundle keeps all of them in FP32.
- The tokenizer is the tokenizers configuration transformers derives from `spiece.model`: no normalizer,
  whitespace split, a prepended and splitting `U+2581`, Unigram segmentation with fused unknown runs, `</s>`.
  The build rejects a different configuration.

## Quantization

`--quantization fp8` and `--quantization nvfp4` quantize the MMDiT of a BF16 bundle at build time; the T5
text encoder and the VAE stay BF16. The default `none` builds exactly as before. Quantized bundles support
image sides from 512 to 1024 (the calibrated range); NVFP4 needs a Blackwell GPU.

```bash
trtmc build lodestones/Chroma1-HD -o chroma1-hd-fp8.bundle --quantization fp8
trtmc build lodestones/Chroma1-HD -o chroma1-hd-nvfp4.bundle --quantization nvfp4 --image-height 512 --image-width 512
```

No ModelOpt FP8 or NVFP4 checkpoint of Chroma1-HD is published, so the BF16 checkpoint is quantized at
build time (`quantization.py`); FP8 and FP4 weights are never stored.

| DiT part | `fp8` | `nvfp4` |
| --- | --- | --- |
| Query, key, value and both feed-forward projections of the image and text streams of the 19 double blocks; query, key, value and the MLP half of `proj_out` of the 38 single blocks | FP8 W8A8 | FP8 W8A8 |
| `proj_mlp` of the 38 single blocks | FP8 W8A8 | NVFP4 W4A4 |
| Attention output projections (`to_out.0`, `to_add_out`, the attention half of `proj_out`) | BF16 | BF16 |
| Embedders, modulation approximator, final layer, norms, attention | BF16 | BF16 |

- FP8: per-tensor e4m3 weights with scale `amax / 448`, static per-tensor activation scales.
- NVFP4: E2M1 weights with one e4m3 scale per 16 inputs and `weight_scale_2 = amax / (6 x 448)` per tensor;
  the activation is block-quantized at run time (`add_dynamic_quantize`, block 16) with a static global
  scale equal to the FP8 activation scale divided by 6.
- Activation scales: `data/chroma1-hd-scales.json`, the maximum of `|input|` of each projection over all
  steps of 32 PartiPrompts at 1024 x 1024 and 512 x 512 (40 steps, guidance 3.0, both guidance branches),
  divided by 448 (`calibrate_scales.py`, a developer script). No projection was above 20x the median of its
  kind, the limit above which a projection would stay BF16. The file is tied to the model config and to the
  checkpoint weights; another checkpoint raises and names the calibration script. A checkpoint that is
  already quantized is refused.
- The single-block `proj_out` reads `[attention output, MLP output]`. TensorRT 11.4 cannot compile a
  Quantize on the fused attention output, so the projection is split by input columns and only its MLP half
  is quantized. That half reads the output of a bias + GELU and stays FP8 in the NVFP4 build.

Speed and size, TensorRT 11.4.0.106, bundles built by the family `build()`, one job per GPU. DiT call: batch
1, 512 text tokens, median of 12 interleaved rounds of 5 calls. Image: 2 text-encoder calls, 80 DiT calls
(40 steps with guidance) and the VAE decoder.

| GPU | Size | Build | DiT plan | Bundle | DiT call | Image | DiT vs `none` | DiT vs `fp8` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B200 | 1024 | `none` | 16.59 GiB | 27.47 GiB | 90.1 ms | 7.34 s | 1.00x | 0.76x |
| B200 | 1024 | `fp8` | 9.25 GiB | 20.12 GiB | 68.0 ms | 5.57 s | 1.32x | 1.00x |
| B200 | 1024 | `nvfp4` | 8.66 GiB | 19.54 GiB | 65.4 ms | 5.36 s | 1.38x | 1.04x |
| B200 | 512 | `none` | 16.59 GiB | 27.46 GiB | 27.7 ms | 2.30 s | 1.00x | 0.73x |
| B200 | 512 | `fp8` | 9.24 GiB | 20.12 GiB | 20.2 ms | 1.70 s | 1.37x | 1.00x |
| B200 | 512 | `nvfp4` | 8.66 GiB | 19.54 GiB | 19.0 ms | 1.60 s | 1.46x | 1.07x |
| B300 | 1024 | `none` | 16.59 GiB | 27.47 GiB | 82.4 ms | 6.70 s | 1.00x | 0.77x |
| B300 | 1024 | `fp8` | 9.25 GiB | 20.12 GiB | 63.4 ms | 5.18 s | 1.30x | 1.00x |
| B300 | 1024 | `nvfp4` | 8.66 GiB | 19.54 GiB | 61.1 ms | 5.00 s | 1.35x | 1.04x |
| B300 | 512 | `none` | 16.59 GiB | 27.46 GiB | 24.8 ms | 2.05 s | 1.00x | 0.78x |
| B300 | 512 | `fp8` | 9.24 GiB | 20.12 GiB | 19.3 ms | 1.61 s | 1.28x | 1.00x |
| B300 | 512 | `nvfp4` | 8.66 GiB | 19.54 GiB | 18.0 ms | 1.50 s | 1.38x | 1.08x |

TensorRT 11.1.0.106 (B200, 512, same job layout): DiT call 27.4 ms `none`, 20.7 ms `fp8`, 19.5 ms `nvfp4`, same
plan sizes, no non-finite output.

Quality: final images of held-out PartiPrompts (not calibration prompts), 40 steps, guidance 3.0, fixed
seeds; the text encoder and the VAE run in PyTorch and only the DiT differs. LPIPS (AlexNet) against the
`none` (BF16) plan; B200 and B300 gave the same numbers at 1024 and 720.

| Size | Prompts | `fp8` LPIPS | `nvfp4` LPIPS | `nvfp4` / `fp8` | PSNR `fp8` / `nvfp4` |
| --- | --- | --- | --- | --- | --- |
| 1024 x 1024 | 24 | 0.103 | 0.151 | 1.47x | 24.9 / 23.2 dB |
| 512 x 512 (B200) | 32 | 0.127 | 0.174 | 1.37x | 23.6 / 21.3 dB |
| 512 x 512 (B300) | 32 | 0.114 | 0.165 | 1.45x | 23.9 / 21.4 dB |
| 720 x 720 | 12 | 0.143 | 0.195 | 1.36x | 24.6 / 21.9 dB |

For scale, a flow-matching sampler turns small numeric differences into visible changes: the BF16 plan
against the PyTorch BF16 pipeline has LPIPS 0.066 (1024) and 0.043 (512), and the same BF16 network built
at builder optimization level 3 instead of 1 has LPIPS 0.045 (1024) and 0.054 to 0.058 (512) against the
level-1 plan.

No non-finite output: 13 stress inputs per build (timesteps 0, 1, 500 and 999; zero latent; latents x4, x10
and x30; text embeddings x0, x4 and x8; an empty prompt; a full text mask) at 512, 720 and 1024 on B200 and
B300, and every DiT call of the quality runs.

Measured and not used (1024 and 512, B200 and B300, TensorRT 11.4): FP8 also on the attention output
projections of the double blocks was 1% to 4% faster with mixed quality (LPIPS 0.128 instead of 0.101 at
1024); FP8 without the feed-forward output projections was 10% to 14% slower; NVFP4 on all 380 projections
was 1.04x to 1.09x faster than FP8 with 1.7x to 2.8x the FP8 LPIPS; NVFP4 on more MLP projections (the MLP
half of `proj_out`, the double-block MLPs) gave 1.4x to 2.7x the FP8 LPIPS for up to about 4% more speed than the
shipped NVFP4.

## Not supported

Image-to-image and inpainting (`ChromaImg2ImgPipeline`, `ChromaInpaintPipeline`), IP-Adapter images, batched
generation, FP16 builds, quantized FP32 builds, a quantized text encoder or VAE, and pixel-space Chroma
variants.
