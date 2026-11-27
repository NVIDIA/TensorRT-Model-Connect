# Qwen-Image 2.1

Text-to-image for diffusers `QwenImage21Pipeline` checkpoints (`Qwen/Qwen-Image-2.1` and mirrors such as
`unsloth/Qwen-Image-2.1`). Task: `text_to_image`. Precision: `bf16`.

The tests pin `unsloth/Qwen-Image-2.1`; its weight, configuration, scheduler and tokenizer files are byte-identical
to `Qwen/Qwen-Image-2.1`. Sources, revisions and licenses of the checkpoint and of the reference implementations
are listed in `NOTICE`.

## What a bundle contains

| Section | Content |
| --- | --- |
| `text_encoder.plan` | Qwen3-VL language model. Emits the last decoder layer's states before the final norm, with the system turn dropped. |
| `denoiser.plan` | `QwenImage21Transformer2DModel`: 32 single-stream blocks over text slots followed by the image's latent tokens. |
| `vae.plan` | `AutoencoderKLQwenImage21` decoder for one image; de-normalises latents in the graph. |
| `tokenizer.json` | The checkpoint's BPE tokenizer. |
| `runtime.json` | Output size, text budget, size of the text encoder's token embedding, prompt frame, scheduler constants. |

Every plan is static. Floating-point engine I/O is FP32, and the two integer inputs are INT32: the text encoder's
`input_ids` and the denoiser's `text_length`. Linear layers, convolutions and attention run in BF16; normalisation
statistics and the transformer's rotary rotation run in FP32, as in the reference.

The graph builders take the compute type as a parameter. Bundles are always built in BF16; the parity tests also
build the graphs in FP32, where they reproduce the reference modules to FP32 accuracy.

## Prompt tokenization

The runtime frames the prompt as the reference pipeline does and tokenizes it with the checkpoint's byte-level BPE
vocabulary. Before the merges it splits the text with the checkpoint's pre-tokenizer pattern,

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

alternative by alternative, with the full Unicode letter and number classes. Merges then apply as in the reference
BPE model: one at a time, the lowest rank first and the leftmost pair among equal ranks. Added tokens are matched
literally, the longest first. The encoding is byte-level because the tokenizer's pre-tokenizer is, whatever tokens
its vocabulary holds.

The build refuses a `tokenizer.json` that configures anything else: another pre-tokenizer or normalizer, the BPE
model options `ignore_merges`, `dropout`, `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `fuse_unk`
or `byte_fallback`, or added tokens with `single_word`, `lstrip` or `rstrip`. When it loads a bundle, the runtime
refuses the same added-token options and the same model options except `end_of_word_suffix`, which it implements,
as well as incomplete added-token records and merge rules that name a token outside the vocabulary.

Token ids index the text encoder's token embedding. The build refuses a tokenizer that gives a token of its
vocabulary, an added token or the padding token an id outside that table, and records the table's size in
`runtime.json`. The runtime refuses a bundle whose `tokenizer.json` or padding id does not fit that size, so the
encoder is never fed an id it has no row for.

The family tests compare the tokenizer with the Hugging Face `tokenizers` library: the split on text of many
scripts, numbers of every kind, quotes, contractions, and whitespace and line-break runs; the merge order on a
merge table for which merging every occurrence of the best pair per pass would give other ids; and the encoding
of a byte-level tokenizer that has U+2581, the SentencePiece space marker, as an added token.

The letter and number classes come from Unicode 16.0 data,
[UnicodeData.txt](https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt) (SHA-256
`ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`), used under the Unicode License v3 in
`UNICODE_LICENSE.txt`, which is installed with the family. They are the classes of the regular-expression engine
in `tokenizers` 0.22.2. Regenerate the table with:

```bash
python families/qwen_image2_1/tools/generate_unicode_categories.py \
  UnicodeData.txt families/qwen_image2_1/runtime/unicode_categories.h
```

## Build geometry

- `image_height` / `image_width`: the output size, default 1024 x 1024. Multiples of 32, at least 256, and at most
  the scheduler's `max_image_seq_len` latent tokens (one token covers 16 x 16 pixels). The cap is this family's: the
  reference extrapolates its schedule shift beyond it, the family does not build there.
- `max_sequence_length`: text slots after the system turn, default 512. The slots hold the prompt and the 8 tokens
  that frame it. A prompt that needs more is rejected at run time. The denoiser recomputes the text rows at every
  step, so a smaller budget is faster.

Another size or budget is a rebuild.

## Denoiser layout

Attention is block-causal: text rows attend causally to text, image rows attend to every valid text key and every
image key. The engine runs this as two attention calls per block. Prompts are padded on the right, so the causal text
call needs no mask; the image call hides padded text slots with a key mask. Text rows are modulated from timestep 0
and image rows from the sampled timestep (`causal_condition`). The key mask and the image block's rotary frame
position both follow from the prompt's token count, which is the engine's `text_length` input.

This is the reference's uncached attention path (`use_kv_cache=False`). Its cached path gives a different BF16
trajectory of the same quality; the family E2E compares against the uncached one.

## Runtime configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `seed` | `0` | Seed of the Gaussian initial latents. |
| `num_steps` | `40` | Denoising steps; at least 2. |
| `cfg_scale` | `1.0` | True classifier-free guidance scale. Applied above 1 together with `negative_prompt`. |
| `negative_prompt` | none | Negative prompt for guidance. |
| `height`, `width` | bundle size | Fixed by the bundle. |

Initial latents, when supplied, are float32 `[64, height / 16, width / 16]`. Output pixels are RGBA.

The denoiser receives the timestep the BF16 reference pipeline feeds its transformer: the sigma rounded through
BF16. Latents stay in FP32 between steps.

## Numerical behaviour

The sampler amplifies rounding differences, so two BF16 implementations of this model do not produce the same
pixels, and neither does the reference in BF16 and FP32.

### One-time precision study

The numbers of this subsection were measured once, when the family was added: on an NVIDIA GB200 with TensorRT
11.4, with scripts that are not part of the repository, on the first revision of the family. No test repeats them.
They describe the engine graphs and the sampler, which that revision and this directory build alike.

Twelve prompt / seed pairs at 512 x 512 and 20 steps (PSNR of the decoded images, minimum / median / maximum):

| Pair | No guidance | Guidance scale 4 |
| --- | --- | --- |
| Reference BF16 against reference FP32 | 25 / 32 / 38 dB | 23 / 27 / 33 dB |
| Bundle against reference BF16 | 20 / 29 / 39 dB | 17 / 27 / 29 dB |
| Bundle against reference FP32 | 20 / 31 / 38 dB | 18 / 27 / 30 dB |

Per denoiser call, on the inputs of the reference BF16 trajectory, the engine and the reference BF16 module are
equally far from the reference FP32 module (about 1.1 % relative error each). Built in FP32, the three graphs
match the reference FP32 modules to 1e-5 per denoiser call and the sampled image to 81 dB (68 dB at guidance
scale 4).

### E2E cases

The family E2E therefore bounds drift rather than asking for equal pixels (PSNR at least 20 dB, windowed SSIM at
least 0.90, layout correlation at least 0.95), and uses prompts and seeds on which the reference agrees with itself
across precisions. Every image is compared at the 8 bits per channel of the PNG the CLI writes; the SDK images and
the reference are float pixels and are rounded the same way first. The layout correlation is the correlation of
the two images' 32 x 32 pixel block means, each colour channel around its own mean, so that it measures where an
image is light and dark and not the overall colour the two share.

The E2E measures these values on every run. Native image against the reference BF16 run, on an NVIDIA B300 with
TensorRT 11.4 and the tests of this revision:

| Case | PSNR | Windowed SSIM | Layout correlation |
| --- | --- | --- | --- |
| `qwen-image-2.1-l0` (512 x 512, 20 steps) | 29.2 dB | 0.985 | 0.9952 |
| `qwen-image-2.1-l0-cfg` (same, guidance scale 4) | 29.3 dB | 0.980 | 0.9992 |
| `qwen-image-2.1` (1024 x 1024, 40 steps) | 41.1 dB | 0.992 | 0.9994 |

For the prompts and seeds of these cases, the one-time study measured the reference BF16 run 34.8, 33.3 and
38.5 dB from the reference FP32 run. Rebuilding the engines on one GPU reproduces the PSNR to 0.01 dB. Another GPU
model can select other kernels and land elsewhere in the spread of the study: on the GB200, the E2E of the family's
first revision scored 28.2, 29.7 and 36.5 dB, against the reference before rounding. Rounding the reference moves
the PSNR of these cases by 0.00 to 0.15 dB; the largest shift belongs to the image with the largest flat area.

## Quantization

`--quantization fp8` and `--quantization nvfp4` quantize the denoiser at build time; the text encoder and the
VAE stay BF16. The default `none` builds exactly the network it built before. NVFP4 needs a Blackwell GPU.

```bash
trtmc build unsloth/Qwen-Image-2.1 -o qwen-image-2.1-fp8.bundle --quantization fp8
trtmc build unsloth/Qwen-Image-2.1 -o qwen-image-2.1-nvfp4.bundle --quantization nvfp4
```

No ModelOpt FP8 or NVFP4 checkpoint of Qwen-Image 2.1 is published, so the BF16 checkpoint is quantized at
build time (`quantization.py`); FP8 and FP4 weights are never stored.

| Denoiser part | `fp8` | `nvfp4` |
| --- | --- | --- |
| Query, key and value projections of the 32 blocks | FP8 W8A8 | FP8 W8A8 |
| SwiGLU input projections (`img_mlp.gate_layer`, `img_mlp.proj`) of the 32 blocks | FP8 W8A8 | NVFP4 W4A4 |
| SwiGLU output projection (`img_mlp.out`) of 30 blocks | FP8 W8A8 | FP8 W8A8 |
| `img_mlp.out` of blocks 5 and 6, attention output projection, text and latent input projections, timestep embedder, shared modulation, output head, norms, attention | BF16 | BF16 |

- FP8: per-tensor e4m3 weights with scale `amax / 448`, static per-tensor activation scales; the dequantized
  operands and the GEMM output are BF16.
- NVFP4: E2M1 weights with one e4m3 scale per 16 inputs and `weight_scale_2 = amax / (6 x 448)` per tensor;
  the activation is block-quantized at run time (`add_dynamic_quantize`, block 16) with a static global
  scale equal to the FP8 activation scale divided by 6.
- Projections that read the same tensor (query, key and value; the two SwiGLU inputs) share one activation
  quantization.
- Activation scales: `data/qwen-image-2.1-scales.json`, the maximum of `|input|` of each projection over all
  steps of 32 PartiPrompts, at 1024 x 1024 (40 steps, no guidance) and at 512 x 512 (20 steps, guidance
  scale 4, both branches), with the pipeline's uncached attention path; divided by 448
  (`calibrate_scales.py`, a developer script). The inputs of `img_mlp.out` in blocks 5 and 6 reach 3360 and
  4960, 26x and 39x the median of that projection (129); a projection above 20x its median stays BF16. The
  file is tied to the model config and to the checkpoint weights; another checkpoint raises and names the
  calibration script. A checkpoint that is already quantized is refused. Other image sizes use the same
  scales and were not measured.

Speed and size, TensorRT 11.4.0.106, bundles built by the family `build()`, one job per GPU. DiT call: the
median of 12 interleaved rounds of 5 calls on a real prompt. Image: one text-encoder call, one DiT call per
step and the VAE decoder, without guidance (guidance doubles the text-encoder and DiT calls).

| GPU | Config | Build | DiT plan | Bundle | DiT call | Image | DiT vs `none` | DiT vs `fp8` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B200 | 1024 x 1024, 512 text slots, 40 steps | `none` | 13.26 GiB | 27.85 GiB | 78.0 ms | 3.15 s | 1.00x | 0.70x |
| B200 | 1024 x 1024, 512 text slots, 40 steps | `fp8` | 7.37 GiB | 21.95 GiB | 54.4 ms | 2.21 s | 1.43x | 1.00x |
| B200 | 1024 x 1024, 512 text slots, 40 steps | `nvfp4` | 6.06 GiB | 20.64 GiB | 50.2 ms | 2.04 s | 1.55x | 1.08x |
| B200 | 512 x 512, 128 text slots, 20 steps | `none` | 13.26 GiB | 27.85 GiB | 17.2 ms | 0.354 s | 1.00x | 0.76x |
| B200 | 512 x 512, 128 text slots, 20 steps | `fp8` | 7.36 GiB | 21.95 GiB | 13.1 ms | 0.272 s | 1.31x | 1.00x |
| B200 | 512 x 512, 128 text slots, 20 steps | `nvfp4` | 6.05 GiB | 20.63 GiB | 11.1 ms | 0.232 s | 1.55x | 1.18x |
| B300 | 1024 x 1024, 512 text slots, 40 steps | `none` | 13.26 GiB | 27.85 GiB | 69.8 ms | 2.82 s | 1.00x | 0.72x |
| B300 | 1024 x 1024, 512 text slots, 40 steps | `fp8` | 7.37 GiB | 21.95 GiB | 50.6 ms | 2.05 s | 1.38x | 1.00x |
| B300 | 1024 x 1024, 512 text slots, 40 steps | `nvfp4` | 6.05 GiB | 20.64 GiB | 46.0 ms | 1.87 s | 1.52x | 1.10x |
| B300 | 512 x 512, 128 text slots, 20 steps | `none` | 13.26 GiB | 27.85 GiB | 16.8 ms | 0.345 s | 1.00x | 0.76x |
| B300 | 512 x 512, 128 text slots, 20 steps | `fp8` | 7.36 GiB | 21.95 GiB | 12.7 ms | 0.264 s | 1.32x | 1.00x |
| B300 | 512 x 512, 128 text slots, 20 steps | `nvfp4` | 6.05 GiB | 20.63 GiB | 10.8 ms | 0.226 s | 1.55x | 1.18x |

TensorRT 11.1.0.106 (B200, 512 x 512, same job layout): DiT call 17.8 ms `none`, 13.6 ms `fp8`, 11.9 ms
`nvfp4`, same plan sizes, LPIPS 0.073 (`fp8`) and 0.127 (`nvfp4`) on 8 prompts, no non-finite output.

Quality: final images of held-out PartiPrompts (not calibration prompts) and the two manifest prompts, fixed
seeds; the text encoder and the VAE run in PyTorch and only the DiT differs. LPIPS (AlexNet) and PSNR against
the `none` (BF16) plan, 24 prompts per row (8 with guidance scale 4).

| Config | GPU | `fp8` LPIPS | `nvfp4` LPIPS | `nvfp4` / `fp8` | PSNR `fp8` / `nvfp4` |
| --- | --- | --- | --- | --- | --- |
| 1024 x 1024, 40 steps | B200 | 0.095 | 0.139 | 1.46x | 25.2 / 22.1 dB |
| 1024 x 1024, 40 steps | B300 | 0.077 | 0.139 | 1.80x | 26.9 / 22.1 dB |
| 512 x 512, 20 steps | B200 | 0.081 | 0.129 | 1.59x | 27.1 / 23.6 dB |
| 512 x 512, 20 steps | B300 | 0.080 | 0.129 | 1.61x | 27.7 / 23.6 dB |
| 512 x 512, 20 steps, guidance 4 | B200 | 0.133 | 0.176 | 1.32x | |
| 512 x 512, 20 steps, guidance 4 | B300 | 0.098 | 0.176 | 1.79x | |

The FP8 GEMMs select other kernels on B200 and B300, hence the two FP8 rows; the NVFP4 images are the same
on both. For scale, the `none` plan against the PyTorch BF16 pipeline (same uncached attention path) has LPIPS
0.008 at 1024 and 0.026 at 512 (8 prompts each).

No non-finite output: 14 stress inputs per build and size (sigma 0, 0.001, 0.5 and 1; zero latent; latents
x4, x10 and x30; text states x0, x4 and x8; an empty prompt; one valid text slot; every text slot valid) on
B200 and B300, and every DiT call of the quality runs.

The E2E cases `qwen-image-2.1-l0-fp8` and `qwen-image-2.1-l0-nvfp4` (512 x 512, 20 steps, seed 7) compare the
native image of a quantized bundle with the same official BF16 run as the BF16 cases. Measured on B200 / B300
with TensorRT 11.4:

| Case | PSNR | Windowed SSIM | Layout correlation | Limits |
| --- | --- | --- | --- | --- |
| `qwen-image-2.1-l0-fp8` | 26.5 / 26.8 dB | 0.976 / 0.971 | 0.998 / 0.998 | 20 dB, 0.90, 0.95 (default) |
| `qwen-image-2.1-l0-nvfp4` | 19.6 / 19.5 dB | 0.921 / 0.921 | 0.942 / 0.940 | 16 dB, 0.85, 0.85 |
| either, every activation scale x0.1 | 7.8 to 7.9 dB | 0.37 to 0.38 | 0.18 to 0.19 | fails |

Seed 7 keeps both quantized images close to the reference on both GPUs; with seed 42 of the BF16 case the
FP8 image scores 17 to 20 dB and the NVFP4 image 15 dB, so one image says little about the format there.

Measured and not used (B200 and B300, TensorRT 11.4, 24 prompts at 1024 and 32 at 512, LPIPS against the
`none` plan):

| Layout | DiT vs `fp8` | LPIPS 1024 / 512 | LPIPS vs `fp8` |
| --- | --- | --- | --- |
| FP8 without `img_mlp.out` | 0.88x to 0.97x | 0.067 / 0.079 | 0.70x to 0.93x |
| FP8 on the three SwiGLU projections only | 0.91x to 0.96x | 0.066 to 0.077 / 0.064 to 0.067 | 0.76x to 0.85x |
| NVFP4 on all six projections | 1.13x to 1.21x | 0.195 / 0.185 | 2.06x to 2.53x |
| NVFP4 on the three SwiGLU projections, attention FP8 | 1.13x to 1.31x | 0.175 / 0.153 | 1.77x to 2.27x |
| Shipped NVFP4, blocks 0, 1, 30 and 31 FP8 | 1.07x to 1.21x | 0.138 to 0.139 / 0.121 to 0.135 | 1.43x to 1.79x |

## Not supported

- Image-conditioned generation (`images_text_to_image_edit`) is not built: it needs the Qwen3-VL vision tower, the
  VAE encoder and condition-image blocks in the denoiser.
- Unicode normalisation of the prompt. The checkpoint's tokenizer normalises text to NFC before tokenizing; the
  runtime tokenizes the prompt as given, so text that is not already NFC is tokenized differently from the reference.
- Checkpoints whose scheduler does not use `num_train_timesteps = 1000`, or that configure `sample_sigmas`.
