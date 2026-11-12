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
| `runtime.json` | Output size, text budget, prompt frame, scheduler constants. |

Every plan is static. Engine I/O is FP32; linear layers, convolutions and attention run in BF16; normalisation
statistics and the transformer's rotary rotation run in FP32, as in the reference.

The graph builders take the compute type as a parameter. Bundles are always built in BF16; the parity tests also
build the graphs in FP32, where they reproduce the reference modules to FP32 accuracy.

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
pixels, and neither does the reference in BF16 and FP32. Measured on twelve prompt / seed pairs at 512 x 512 and 20
steps (PSNR of the decoded images, minimum / median / maximum):

| Pair | No guidance | Guidance scale 4 |
| --- | --- | --- |
| Reference BF16 against reference FP32 | 25 / 32 / 38 dB | 23 / 27 / 33 dB |
| Bundle against reference BF16 | 20 / 29 / 39 dB | 17 / 27 / 29 dB |
| Bundle against reference FP32 | 20 / 31 / 38 dB | 18 / 27 / 30 dB |

Per denoiser call, on the inputs of the reference BF16 trajectory, the engine and the reference BF16 module are
equally far from the reference FP32 module (about 1.1 % relative error each). Built in FP32, the three graphs
match the reference FP32 modules to 1e-5 per denoiser call and the sampled image to 81 dB (68 dB at guidance
scale 4).

The family E2E therefore bounds drift rather than asking for equal pixels (PSNR at least 20 dB, windowed SSIM at
least 0.90, layout correlation at least 0.95), and uses prompts and seeds on which the reference agrees with itself
across precisions. Its cases, native CLI image against the reference BF16 run, measured on an NVIDIA GB200 with
TensorRT 11.4:

| Case | PSNR | Windowed SSIM | Layout correlation | Reference BF16 against reference FP32 |
| --- | --- | --- | --- | --- |
| `qwen-image-2.1-l0` (512 x 512, 20 steps) | 28.2 dB | 0.982 | 0.994 | 34.8 dB |
| `qwen-image-2.1-l0-cfg` (same, guidance scale 4) | 29.7 dB | 0.981 | 0.999 | 33.3 dB |
| `qwen-image-2.1` (1024 x 1024, 40 steps) | 36.5 dB | 0.987 | 0.998 | 38.5 dB |

Rebuilding the engines on that GPU reproduces these numbers to 0.01 dB. Another GPU model can select other kernels
and land elsewhere in the spread above.

## Not supported

- Image-conditioned generation (`images_text_to_image_edit`) is not built: it needs the Qwen3-VL vision tower, the
  VAE encoder and condition-image blocks in the denoiser.
- Unicode normalisation of the prompt. The checkpoint's tokenizer normalises text to NFC before tokenizing; the
  runtime tokenizes the prompt as given, so text that is not already NFC is tokenized differently from the reference.
- Exact pre-tokenization of uncommon text. The pre-tokenizer, copied from the `qwen_image` family, approximates the
  reference's Unicode letter and number classes (for example, fullwidth digits are not treated as digits) and two of
  its quote rules: a quote directly before letters that do not form a contraction, and upper-case contractions such
  as `DON'T`. Prompts with such text can be tokenized differently from the reference.
- Checkpoints whose scheduler does not use `num_train_timesteps = 1000`, or that configure `sample_sigmas`.
