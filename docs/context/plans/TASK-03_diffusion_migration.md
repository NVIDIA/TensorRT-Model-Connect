# TASK-03: Migrate FluxPipeline + WanPipeline + ZImagePipeline to TrtModule

## Branch: `agent-X-migrate-diffusion`

## Goal

Replace `FluxDiffusionBackend`, `WanDiffusionBackend`, and `ZImageDiffusionBackend` delegation with direct `TrtModule::forward()` calls. No KV cache needed — diffusion uses a scheduler loop.

## Current State

```
FluxPipeline   → FluxDiffusionBackend   → text_encoders + denoiser + VAE (raw TRT)
WanPipeline    → WanDiffusionBackend    → text_encoder + denoiser + VAE (raw TRT)
ZImagePipeline → ZImageDiffusionBackend → text_encoder + denoiser + VAE (raw TRT)
```

## Target State

```
FluxPipeline   → TrtModule(T5) + TrtModule(CLIP) + TrtModule(denoiser) + TrtModule(VAE)
                  + IScheduler + preprocessor weights
WanPipeline    → TrtModule(T5) + TrtModule(denoiser) + TrtModule(VAE)
                  + IScheduler + preprocessor weights
ZImagePipeline → TrtModule(Qwen3) + TrtModule(denoiser) + TrtModule(VAE)
                  + IScheduler + preprocessor weights
```

## Shared Diffusion Pattern

All three backends follow the same structure:

1. **Text encoding**: tokenize prompt → `text_encoder->forward({input_ids})` → text embeddings
2. **Latent init**: random noise in latent space
3. **Scheduler loop**: for each timestep:
   - Pack latents (2x2 spatial packing for FLUX, 3D for Wan)
   - Embed hidden states (x_embedder matmul)
   - Compute timestep embedding
   - `denoiser->forward({hidden, encoder_hidden, timestep_emb, rope})` → velocity
   - Unpack velocity
   - Scheduler step: `latents += dt * velocity`
4. **VAE decode**: `vae->forward({latents})` → pixel image
5. **Post-process**: clamp to [0,1], convert to HWC

### Model-specific differences:

| Aspect | FLUX | Wan | Z-Image |
|--------|------|-----|---------|
| Text encoder | T5 + CLIP (dual) | T5 (single) | Qwen3 (single) |
| Latent channels | 16 (FLUX.1) / 32 (FLUX.2) | 16 | 16 |
| Spatial packing | 2x2 patch → tokens | 3D video patches | 2x2 patch |
| Scheduler | Flow-match Euler | Flow-match Euler | Flow-match Euler |
| RoPE | 2D positional | 3D (temporal+spatial) | 2D positional |
| Preprocessor | x_embedder + context_embedder | patch_embed_3d | x_embedder + cap_proj |
| Output | Single image | Multi-frame video | Single image |
| FLUX.2 special | BN denorm + unpatchify before VAE | — | — |

## Steps for Each Pipeline

### FluxPipeline (M-12, ~600 LOC)
1. Port T5 text encoding as `t5_module->forward({input_ids})`
2. Port CLIP encoding as `clip_module->forward({input_ids})` (optional, for pooled output)
3. Port latent initialization + 2x2 packing/unpacking functions
4. Port denoiser loop: `denoiser->forward({hidden, encoder_hidden, temb, cos, sin})`
5. Port VAE decode: `vae->forward({latents})`
6. Handle FLUX.1 vs FLUX.2 differences (latent layout, scheduler mu, BN denorm)

### WanPipeline (M-13, ~500 LOC)
1. Port T5 text encoding
2. Port 3D video latent initialization + patch embedding
3. Port denoiser loop with 3D RoPE
4. Port VAE decode (video frames)

### ZImagePipeline (M-14, ~500 LOC)
1. Port Qwen3 text encoding
2. Port latent init + caption projection
3. Port denoiser loop
4. Port VAE decode
5. Port Z-Image-specific preprocessor weight loading

## Files to Modify
- `src/runtime/pipelines/diffusion_pipeline.h/cpp`
- `src/runtime/pipeline_factory.cpp`

## Files to Delete (after verification)
- `src/runtime/trt/diffusion/flux_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/wan_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/z_image_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/diffusion_backend_base.cpp/h`
- `src/runtime/trt/diffusion/diffusion_backend.h`
- `src/runtime/pipelines/diffusion_backend_factory.cpp/h`

## Verification
```bash
pytest tests/test_e2e.py::test_e2e[flux-schnell] tests/test_e2e.py::test_e2e[flux-2-dev] tests/test_e2e.py::test_e2e[wan21-t2v-1.3b] tests/test_e2e.py::test_e2e[z-image-turbo] tests/test_e2e.py::test_e2e[pixart-sigma-1024] -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

## Independent of TASK-01/02
Can be done in parallel with audio migration since diffusion and audio share no code.
