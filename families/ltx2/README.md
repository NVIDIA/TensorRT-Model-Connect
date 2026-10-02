# LTX-2.5 (`ltx2`)

Text-to-audio-video for Lightricks LTX-2.5 diffusers checkpoints (`LTX2Pipeline`, for example
`Lightricks/LTX-2.5-Diffusers`). One bundle generates a video and its 48 kHz stereo soundtrack.

## Scope

- Task `text_to_audio_video`, precision `bf16`, batch 1.
- The distilled transformer with its 8-step schedule (guidance 1, no STG). The bundle bakes
  the schedule into `runtime.json`. The full transformer (30 steps, CFG/STG/modality
  guidance) is not supported yet.
- The video size and frame count are fixed at build time (`--image-height`, `--image-width`,
  `--video-num-frames`). Height and width must be multiples of 32, and the frame count must
  be `8n+1`. The default is 960x544, 121 frames at 24 fps.
- `--context-parallel-size 1` runs on one GPU. `--context-parallel-size 2` splits the
  video tokens of the DiT across two GPUs. The text encoder runs on every rank. The video
  VAE decodes in tiles that the ranks share, and the last rank decodes the audio
  (see [Tiled video decode](#tiled-video-decode)).

## Bundle

| Section | Contents |
|---|---|
| `text_encoder.plan` | Gemma 4 text tower and the LTX-2 text connectors (video and audio context) |
| `denoiser.plan` | Joint audio/video DiT. With CP=2, one plan serves both ranks. |
| `vae.plan` | Video VAE decoder for one tile shape (whole video with `--vae-tile-pixels 0 --vae-tile-frames 0`) |
| `audio.plan` | Audio VAE decoder and vocoder with bandwidth extension |
| `tokenizer.json`, `runtime.json` | Tokenizer, shapes and schedule |

Context parallelism keeps the audio stream and text replicated and shards the video
tokens. Video self-attention all-gathers each rank's normed and rotated keys and values.
Video-to-audio attention merges per-rank softmax statistics through one small all-gather.
The network uses no all-to-all collective.

## Tiled video decode

The video VAE decodes the latent video as overlapping tiles of one shape, so one static plan serves
every tile. The tiles are blended with linear ramps over their overlaps and normalized by the summed
weights, as in the Lightricks and TensorRT-LLM `tiled_decode`. Tiles are at most 512 pixels with at
least 64 pixels of overlap; clips longer than 257 frames also split in time, into tiles of up to
256 frames that overlap by at least 24 frames. `families/ltx2/vae_tiling.py` computes the tile plan
and writes it into `runtime.json`.

- With context parallelism, the ranks decode disjoint tiles. The worker ranks send their decoded
  tiles to rank 0 over NCCL point-to-point on the engines' communicator, and rank 0 blends every
  tile in tile order. The blended video is the same bit for bit as the single-GPU tiled decode.
- The last rank decodes the audio while rank 0 decodes its tiles. On one GPU, the host blend runs
  while the GPU decodes the audio.
- A transfer that does not finish within 10 minutes aborts the NCCL communicator instead of
  leaving NCCL kernels running.

`trtmc ltx2 build` (`python -m tensorrt_model_connect ltx2 build`) takes the tile options
`--vae-tile-pixels`, `--vae-tile-overlap-pixels`, `--vae-tile-frames` and
`--vae-tile-overlap-frames`. A size of 0 leaves that axis untiled, and setting both sizes to 0
builds the untiled decoder. The shared `trtmc build --family ltx2` uses the defaults.

## Build and run

```bash
python -m tensorrt_model_connect build /models/LTX-2.5-Diffusers --family ltx2 \
  --backend trt_rtx --precision bf16 \
  --image-width 960 --image-height 544 --video-num-frames 121 \
  --context-parallel-size 2 -o ltx25-cp2.bundle

trtmc generate-video ltx25-cp2.bundle --runtime-root <trtmc runtime root> \
  --prompt "A red fox walking through a snowy forest at dawn" \
  --output out --set seed=42
```

`generate-video` writes `out/frame-NNNNNN.png` and `out/audio.wav`. Set `TRTMC_LTX2_PROGRESS=1`
to print one progress line per stage and denoising step.

## TensorRT-RTX

- Context parallelism with `--backend trt_rtx` requires TensorRT-RTX 1.7.1 or newer.
  TensorRT-RTX 1.6.x has no multi-device support, so the build stops with an error
  for `--context-parallel-size 2` when an older version is installed.
- Build and run with the same TensorRT-RTX version. A bundle's plans are specific to the
  version that built them.
- TRTMC does not pin `tensorrt-rtx` in `pyproject.toml`. Install the Python package that
  matches your TensorRT-RTX runtime.

## Native Windows with two GPUs

- Put both GPUs in TCC mode. Under WDDM or MCDM they report no peer access and NCCL has
  no transport.
- Build `nccl.dll` from source and point the runtime at it with `TRTMC_NCCL_LIBRARY`.
  See [Multi-Device Execution](../../website/docs/features/multi-device.md).
- Start the ranks with `tools/launch_ranks.py`, because native Windows has no `mpirun`:

```powershell
python tools\launch_ranks.py -n 2 --gpus 0,1 --nccl-library C:\nccl\install\bin\nccl.dll `
  --library-dir <TensorRT-RTX bin directory> `
  -- trtmc generate-video ltx25-cp2.bundle --runtime-root <trtmc runtime root> `
     --prompt "..." --output out --set seed=42
```

To give each rank its own TensorRT-RTX runtime cache, put `{rank}` in the path, for example
`--runtime-cache cache-rank{rank}.bin`.

## Tests

- `tests/test_*_parity.py` build each engine from tiny random weights and compare it
  with diffusers.
- `tests/test_context_parallel.py` runs the CP=2 DiT on two GPUs (torch-free ranks) against
  the single-device plan and diffusers. It also runs the tile-parallel VAE decode and checks that
  it matches the single-GPU tiled decode bit for bit. It needs `TRTMC_NCCL_LIBRARY`.
- `tests/test_vae_tiling.py` checks the tile plan (coverage, ramps, rank assignment) without a GPU.
- `tests/test_e2e.py` builds the real checkpoint and runs the native CLI. It compares the
  output with `LTX2Pipeline` started from the same noise. Select it with `--e2e-model ltx2`.

### E2E thresholds

Both sides run the 48-layer DiT in bf16. A single forward of the native engine differs from
an fp32 diffusers forward by about 3% relative L2, as does diffusers in bf16. The 8 distilled
steps amplify that difference into detail-level changes of the same scene. Exact pixel
agreement is therefore not expected. The thresholds separate "same trajectory" from
"different or broken output". They were set from these measurements on 2x RTX PRO 6000 with
TensorRT-RTX 1.7.1:

| Comparison (seed 42) | Frame PSNR mean / min | Log-spectrogram corr | RMS ratio |
|---|---|---|---|
| Native vs diffusers bf16, 384x640x49 (CP=1) | 19.1 / 18.4 dB | 0.908 | 1.00 |
| Native vs diffusers bf16, 960x544x121 (CP=2) | 18.2 / 16.2 dB | 0.892 | 1.13 |
| Diffusers bf16 vs diffusers fp32 (both sizes) | 22.7-23.8 / 21.9-22.9 dB | 0.92 | 1.00 |
| Diffusers bf16, seed 42 vs seed 7 (unrelated output) | 11.5-12.9 / 10.1-12.2 dB | 0.75-0.81 | 0.25-5.9 |

The thresholds are a mean of at least 15 dB, a minimum of at least 14 dB, a correlation of
at least 0.85 and an RMS ratio in [0.8, 1.25]. Each one sits between the measured native
values and the unrelated-output values. The audio correlation is lower than the video
agreement because the released pipeline runs its vocoder in bf16; against an fp32 vocoder,
the native audio reaches 0.95-0.97.
