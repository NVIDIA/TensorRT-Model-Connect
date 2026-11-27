# Stable Audio 3

Text-to-audio for the post-trained Stable Audio 3 latent diffusion models
(`diffusion_objective: rf_denoiser`) published in the native
[`stable-audio-3`](https://github.com/Stability-AI/stable-audio-3) layout, written
for [`stabilityai/stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium).
The family implements the `text_to_audio` Task and returns 44.1 kHz stereo PCM.

## Pipeline

The bundle holds three engines, the T5Gemma tokenizer and `runtime.json`:

| Engine | Computes | Precision |
|---|---|---|
| `conditioner.plan` | T5Gemma encoder (256 prompt tokens, learned padding embedding) and the `seconds_total` embedding | T5Gemma in BF16, the dtype the checkpoint stores and the reference runs; the embedder in the bundle precision |
| `dit.plan` | one DiT step: velocity for latents `[1, 256, T]`, time, conditioning and the number of valid latents | bundle precision, FP32 for RMSNorm, rotary angles and Fourier features |
| `decoder.plan` | SAME decoder: latents `[1, 256, T]` to audio `[1, 2, 4096 T]` | bundle precision, FP32 for the rotary angles |

The runtime sizes the render like `StableAudioModel.generate` (duration plus 6 s
of headroom, aligned to 2 latents, capped at 120 s), runs the sampler on the
host (pingpong sampler, LogSNR-shifted schedule), masks the headroom, clamps to
[-1, 1] and cuts the output to the requested duration.

## Use

```bash
python -m tensorrt_model_connect build --model stabilityai/stable-audio-3-medium \
  --output stable-audio-3-medium.bundle --precision fp16
trtmc generate-audio stable-audio-3-medium.bundle --runtime-root <build> \
  --prompt "Gentle rain on a tin roof with distant rolling thunder" \
  --set duration=10 --set steps=8 --seed 42 --output rain.wav
```

Task config fields: `duration` (seconds, default 120), `steps` (default 8),
`seed` (default -1 = random). Builds accept `fp16` (default, the reference's
`model_half=True`) and `fp32`; one prompt per call.

## Behaviour that follows the reference

- Medium requires Flash Attention 2 upstream. The DiT therefore hides padded
  latents from every query exactly (varlen semantics), and the decoder attends
  within one latent group on each side (`window_size=(17, 17)`).
- The reference loads T5Gemma with Transformers' default `sdpa` attention, which
  does not apply `attn_logit_softcapping`; the conditioner engine does not either.
- The decoder is deterministic. The official `generate()` adds noise in the
  decoder at inference (bottleneck `noise_regularize`, decoder `mask_noise`); the
  family leaves both out on purpose, as Stability AI's own TensorRT and MLX ports
  of the SAME decoder do, and the E2E oracle switches them off the same way.
- FP16 bundles follow the reference's `model.to(torch.float16)`: parameters and
  buffers are rounded to FP16 (also those the FP32 islands use, such as the
  rotary frequencies), and FP16 GEMMs (linear layers) accumulate in FP32 as the
  reference's cuBLAS GEMMs do. Attention runs through TensorRT's IAttention in
  the bundle precision.
- Gaussian noise comes from a family-owned counter-based generator
  (`runtime/sampling.h`, twin in `tests/family_sampling.py`), so a seed gives the
  same audio on every platform and the official pipeline can be fed the same noise.

Not supported: base checkpoints (`rectified_flow`, such as `medium-base`: they
are sampled with classifier-free guidance and negative prompts), the inpainting
and audio-to-audio heads (they need Task contracts for masked and conditioning
audio), LoRA adapters, batches.

## Validation

`tests/test_e2e.py` builds the bundle, renders through the native CLI and the C
and C++ SDK consumers, and compares the audio with the official pipeline
(`tests/official_reference.py`) run on the same prompt, duration, steps and
noise: waveform SNR and multi-resolution log-spectrogram correlation. Eight
sampler steps amplify small numeric differences: running only the official
T5Gemma in FP32 instead of BF16 already leaves the official audio 21-35 dB SNR
from itself. The floors therefore sit between the measured native-vs-official
agreement and runs with another seed or prompt (`test_e2e.py` records the
calibration). It needs
`TRTMC_BINARY`, `TRTMC_RUNTIME_ROOT`, `TRTMC_NATIVE_BUILD_DIR`, a checkout of the
reference at the revision in `tests/reference-source.json` in
`TRTMC_REFERENCE_SOURCE_DIR`, and the checkpoint (cached, or
`TRTMC_STABLE_AUDIO_3_MODEL_DIR`). The checkpoint is gated on the Hub.
