# flux3

Text-to-image for [`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein)
(FLUX 3 Image Klein, the step-distilled flow-map checkpoint: 16 steps, no guidance), Task `text_to_image`.

The checkpoint is gated on the Hugging Face Hub under the FLUX Kommunity License; accept it there before downloading.
The license allows non-commercial use; commercial use of outputs is limited to Qualifying Users (individuals or
companies under US$5M annual revenue with affiliates) who filter content or review outputs and disclose AI generation
where the law requires it; this family contains no content filter. A bundle built from the checkpoint is a Derivative
whose distribution needs the license's attribution notice. `NOTICE` summarises the terms; the license text governs.

The family reads the Diffusers layout of the repository (`transformer/`, `text_encoder/`, `vae/`,
`modular_model_index.json`); the root single-file package and the `variants/` quantized transformers are not used.

## Bundle

| Section | Content |
|---|---|
| `text_encoder.plan` | Qwen3-VL-4B language layers 0-31 (no vision tower): `input_ids [1, L]`, `valid_length [1]` → `text_embeds [1, L, 20480]` (hidden states after layers 4, 8, …, 32, concatenated per token) |
| `denoiser.plan` | Flux3 transformer, image and text modes: `latents [1, N, 128]`, `text_embeds`, `timestep_proj [1, 256]`, `target_timestep_proj [1, 256]` → `velocity [1, N, 128]` |
| `vae.plan` | latent unpack, batch-norm de-normalisation, 2 x 2 unpatchify, `AutoencoderKLFlux2` decoder → `image [1, 3, H, W]` |
| `tokenizer.json` | the checkpoint's Qwen2 byte-level BPE tokenizer |
| `runtime.json` | geometry, prompt frame, text window, schedule |

Geometry is fixed per bundle (`--image-height` / `--image-width`, multiples of 16, default 1024 x 1024,
N = H/16 · W/16). The text window is dynamic: a prompt is framed as a user turn of the checkpoint's chat template,
padded to the next multiple of 80 tokens and must fit `--max-sequence-length` (a multiple of 80, default 4000,
at most 8192); longer prompts are rejected, not truncated.

## Numerics

The graphs follow the Diffusers `Flux3DistilledModularPipeline` (the prompt upsampler removed): BF16 weights and
activations, QK RMSNorm and RoPE in float32, LayerNorm and RMSNorm statistics in float32, the VAE decoder in float32.
The runtime reproduces the pipeline's BF16 rounding of the timesteps fed to the model and of the latents between
steps, the constant timestep shift 6.93 and the flow-map Euler update.

## Request

`generate-image` / the `text_to_image` SDK Task: a prompt, optional `initial_latents` (float32, `[128, H/16, W/16]`,
the layout of the reference's noise), config `seed` (used when no latents are given) and `num_steps` (default 16).

The model is trained on full scene captions. The official pipeline expands short requests with a separate 9 B prompt
upsampler (`black-forest-labs/flux-3-image-klein-pup`); this family takes the caption as given. Not supported:
image editing and multi-reference composition (`images_text_to_image_edit`), batching, guidance, the quantized
transformer variants, Unicode normalisation of the prompt (the reference tokenizer applies NFC), and the
reference CLI's safety classifier.
