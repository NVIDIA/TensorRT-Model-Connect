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

The transformer graphs preserve the checkpoint's BF16 normalization, dense,
LoRA, activation and attention boundaries through family-owned LibTorch
bindings on the public TVM-FFI interface. Timestep features and projections
retain their FP32 arithmetic. The native runtime registers these bindings
before loading transformer engines and reuses a managed execution stream across
blocks so temporary allocation caches remain bounded. Building and running
requires matching LibTorch CUDA libraries and the TVM-FFI bridge. The FP32 media decoders use
TensorRT operations.

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

The direct tests use the Diffusers commit pinned by the reference requirements.
With `TRTMC_PIKA_H3_FLEX_MODEL_DIR` unset, they load the cached checkpoint at the
manifest revision. An explicit override uses the given directory and records
its revision as unverified. The CPU-generator and operator comparisons use PyTorch 2.12.0;
full generation accuracy remains unqualified. Native Torch must use the same C++
ABI as the runtime toolchain; configuration rejects an incompatible distribution.
The oracle keeps the complete FP32 audio VAE on CUDA because its weight-normalized
convolutions do not support group offloading. It loads the other neural components
one at a time and stages their weights in a temporary directory using Diffusers
disk-backed leaf offloading. This bounds resident host weights and requires
scratch disk space for the offloaded components. It trades transfer time for
lower host and device memory use. The original model, dtypes, scheduler, prompts,
seeds and comparison gates remain the same.

Prepare the manifest's isolated reference environment through the repository's
setup step; the original Diffusers and Transformers pins remain in
`tests/requirements.txt` while the builder uses the normal package environment.

Set `TRTMC_BINARY`, `TRTMC_RUNTIME_ROOT` and
`TRTMC_NATIVE_BUILD_DIR` (the CMake build directory containing the family test
tools), build those tools, then run:

```sh
PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/pika_h3_flex/tests --e2e-model pika_h3_flex -q
```

Checkpoint weights, adapters and generated engines are not distributed here.
Obtain them under their published licenses, including the MiniMax H3 Community
License terms.
