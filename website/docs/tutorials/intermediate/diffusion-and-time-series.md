---
title: Intermediate Tutorial - Diffusion and Vision Pipelines
---

This tutorial covers pipelines that do not return generated text.

These pipelines still use `IPipeline`, but their task methods and internal loops differ.

```mermaid
flowchart LR
  TextTask["Text generation"] --> TokenLoop["token loop"]
  DiffTask["Diffusion"] --> DenoiseLoop["denoising loop"]
  SegmentTask["Segmentation/detection"] --> VisionPost["vision postprocess"]
```

## FLUX image generation

```bash
./build/trtmc build black-forest-labs/FLUX.2-dev \
  -o /tmp/flux2.trtfb \
  --precision fp16 \
  --image-height 1024 \
  --image-width 1024 \
  --num-inference-steps 28

./build/trtmc generate-video /tmp/flux2.trtfb \
  --prompt "A photo of a cat sitting on a windowsill at sunset" \
  --output /tmp/flux2-frames \
  --num-steps 28
```

The command is named `generate-video` because the C++ API returns an `ImageResult` with `num_frames`; single-frame image generation is the same surface as video generation.

```mermaid
flowchart TD
  Prompt["Prompt text"] --> TextEnc["Text encoder"]
  TextEnc --> Conditioning["Conditioning embeddings"]
  Noise["Initial latent noise"] --> Denoiser["Denoiser engine"]
  Conditioning --> Denoiser
  Denoiser --> Scheduler["Scheduler step"]
  Scheduler --> Denoiser
  Scheduler --> Latents["Final latents"]
  Latents --> VAE["VAE decoder"]
  VAE --> Frames["ImageResult pixels/frames"]
```

Diffusion inference is iterative like text generation, but the loop is over denoising steps rather than generated tokens.

| Text generation | Diffusion |
| --- | --- |
| Generates one token per decode step. | Updates a latent tensor each denoising step. |
| Uses KV cache to avoid recomputing prompt attention. | Uses a scheduler to control the noise trajectory. |
| Samples from logits. | Decodes final latents into pixels. |
| Stops at EOS or token budget. | Stops after configured denoising steps. |

## Wan video generation

```bash
./build/trtmc build Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  -o /tmp/wan21.trtfb \
  --precision fp16 \
  --video-height 480 \
  --video-width 832 \
  --video-num-frames 81

./build/trtmc generate-video /tmp/wan21.trtfb \
  --prompt "A cinematic shot of clouds moving over mountains" \
  --output /tmp/wan21-frames
```

Diffusion bundles often have multiple engine sections: text encoder, denoiser, and VAE decoder.

Video generation adds temporal dimensions. The output is still `ImageResult`, but `num_frames` is greater than one and the latent tensor includes time.

## Segmentation and detection mental model

Segmentation and detection are covered by feature and E2E docs, but the runtime shape is similar to other vision pipelines:

```mermaid
flowchart LR
  Pixels["Image pixels"] --> Preprocess["Resize/normalize/layout"]
  Preprocess --> Engine["Vision engine"]
  Engine --> Post["Decode masks or boxes"]
  Post --> Result["SegmentResult or detection JSON"]
```

Current segmentation owners use qualified strategies: SegFormer emits
`segformer_segmentation`, while SAM and SAM3 emit
`sam_prompted_segmentation` and `sam3_prompted_segmentation`. The public API
and CLI also expose `detect()`, but the current model descriptors do not claim
an object-detection runtime strategy. Do not present that API surface as a
supported model until a model-owned descriptor and E2E manifest provide the
evidence.

## What you should understand now

- Not every model returns text.
- `IPipeline` provides task-specific methods so users do not manipulate raw engine tensors.
- Iterative inference can mean token decode, diffusion denoising, streaming audio chunks, or another task-specific loop.
- E2E manifests are the safest way to find canonical inputs for non-text models.
