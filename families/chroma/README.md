# Chroma

`ChromaPipeline` checkpoints (`lodestones/Chroma1-HD`): T5-XXL text encoder, the 8.9 B
`ChromaTransformer2DModel` MMDiT and the FLUX.1 VAE, served as the `text_to_image` Task.

## Bundle

`trtmc build <checkpoint> -o chroma.bundle [--precision bf16|fp32] [--image-height H --image-width W]
[--max-sequence-length L]` builds three TensorRT plans from the checkpoint, the tokenizer and
`runtime.json`. BF16 (as stored) is the default; FP32 is the parity build. The image size (multiples of
16 up to 4096, default 1024 x 1024) and the text length (at most 512, default 512) are fixed per bundle.

## Runtime Config

| Field | Default | Meaning |
| --- | --- | --- |
| `negative_prompt` | `""` | classifier-free guidance prompt, used when `guidance_scale > 1` |
| `num_inference_steps` | 35 | flow-matching Euler steps (1 to 1000) |
| `guidance_scale` | 5.0 | classifier-free guidance scale |
| `seed` | 0 | noise seed; not combined with initial latents |
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

## Not supported

Image-to-image and inpainting (`ChromaImg2ImgPipeline`, `ChromaInpaintPipeline`), IP-Adapter images, batched
generation, FP16 and quantized builds, and pixel-space Chroma variants.
