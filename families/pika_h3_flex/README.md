# Pika H3 Flex

This family implements `text_to_audio_video` with the released FL2VA rank-128
adapter. It owns the Qwen3-VL conditioner through hidden state 50, both text
refiners, all 50 joint denoiser blocks, the six-step stochastic sampler, the
complete video ViT decoder and the alias-free BigVGAN audio decoder.

The supported profile requests 120 frames at 832×480. As upstream specifies,
the frame count is aligned to 124 frames at 24 fps. Output includes the actual
frame timestamps and 165333 interleaved stereo audio frames at 32 kHz, starting
at time zero. The build uses BF16 text/denoiser computation and the published
mixed FP32 projection boundaries, with FP32 video and audio decoders.

`seed` selects the portable CPU-generator stream, defaulting to 42. The runtime
draws video noise, audio noise, then fresh video/audio noise for each of the six
sampling steps. Tests compare this stream to the upstream CPU generator before
comparing full generated video and stereo PCM. The provided sigma grid is
shifted independently by 12 for video and 3 for audio.

The family loads engines in phases. Text blocks are consumed once and released;
each sampling step loads and releases one denoiser block at a time to bound
live weights and execution workspace. Video and audio decoders load afterward.
It returns one completed synchronized clip per request.
Keyframes, references, batching, quantization and larger profiles are outside
this bring-up.

The direct tests use the exact checkpoint and the Diffusers commit and package
versions pinned by its published examples. Set `TRTMC_PIKA_H3_FLEX_MODEL_DIR`,
`TRTMC_BINARY` and `TRTMC_RUNTIME_ROOT`, build the family test tools, then run:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/pika_h3_flex/tests --e2e-model pika_h3_flex -q
```

Checkpoint weights, adapters and generated engines are not distributed here.
Obtain them under their published licenses, including the MiniMax H3 Community
License terms.
