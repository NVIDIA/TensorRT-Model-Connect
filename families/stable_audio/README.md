# Stable Audio Open (`stable_audio`)

Text-to-audio with the diffusers `StableAudioPipeline` checkpoints, on the existing
`text_to_audio` Task: a prompt in, 44.1 kHz stereo float PCM out.

| Checkpoint | Revision | Parameters (stored) |
| --- | --- | --- |
| [`stabilityai/stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0) | `f21265c1e2710b3bd2386596943f0007f55f802e` | 1.21 B DiT + Oobleck + projection, 0.11 B T5-base encoder, F32 |

The repository is gated: accept the Stability AI Community License on its page before
downloading it. The license is not Apache-2.0; read its terms before using the weights.

## Bundle

Three TensorRT engines, built with the network API from the checkpoint's safetensors:

| Section | Network | Precision |
| --- | --- | --- |
| `conditioner.plan` | T5 encoder, text projection, padding mask, start / end duration conditioners | FP32 always (runs once per prompt; T5 is prone to FP16 overflow) |
| `transformer.plan` | `StableAudioDiTModel`, batch 1-2 (the guidance branches) | `fp32`, or `fp16`: FP16 projection / attention operands, projections accumulated in FP32 (FP16-rounded weights and FP16 activations multiplied as FP32 on TF32 tensor cores, which is exact; TensorRT's FP16 tactics may accumulate in FP16); residual stream, layer norms, rotary embedding, timestep embedding and the 1x1 pre / post convolutions in FP32 (without TF32 rounding) |
| `vae_decoder.plan` | `AutoencoderOobleck` decoder, weight norm and Snake parameters folded | `fp32`, or `fp16`: FP16 activations, convolutions accumulated in FP32 like the denoiser's projections |

plus `tokenizer.json` (the checkpoint's T5 tokenizer, used as is) and `runtime.json`
(shapes, special token ids, duration range and the scheduler configuration).

The FP16-rounded weights are stored as FP32 constants: the FP16 denoiser plan is 3.4 GB
(2.1 GB with FP16 storage) and its matrix multiplies run at the TF32 rate.

```bash
trtmc build stabilityai/stable-audio-open-1.0 --revision f21265c1e2710b3bd2386596943f0007f55f802e \
  --precision fp16 -o stable-audio-open-1.0.bundle
trtmc generate-audio stable-audio-open-1.0.bundle --prompt "The sound of a hammer hitting a wooden surface." \
  --set "negative_prompt=Low quality." --set audio_end_in_s=10 --set num_inference_steps=100 --seed 0 \
  --output hammer.wav
```

Build options: `--precision fp32|fp16`, backend `trt`. The text length is the
tokenizer's padded length (128 for this checkpoint); quantization, batching, tensor or
context parallelism and KV-cache options are rejected.

## Request options

The Task's config fields (`--set NAME=VALUE` on the CLI, `--seed` for `seed`):

| Config field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `negative_prompt` | string | absent | Prompt of the unconditional guidance branch. Absent: that branch is all zeros, as in diffusers |
| `audio_start_in_s` | float | 0.0 | Start of the returned audio, and the start-time conditioning |
| `audio_end_in_s` | float | the model window (47.55 s) | End of the returned audio, and the total-time conditioning |
| `num_inference_steps` | int | 100 | Denoising steps, 1-1000 |
| `guidance_scale` | float | 7.0 | Classifier-free guidance; 1.0 or less runs one branch |
| `seed` | int | 0 | Seed of the family noise generator |

Like diffusers, the pipeline always denoises the whole latent window (1,024 frames,
47.55 s) and returns samples `[int(start * 44100), int(end * 44100))`, clipped to the
window. Start and end must lie in the duration conditioners' range [0, 512] s, end must
exceed start, `end - start` must not exceed the window, and start must lie inside the
window (below 47.55 s): a request that would return no samples is rejected, where
diffusers returns empty audio. `guidance_scale` must be a finite float32 value.

## Sampling and noise

The sampler is diffusers' `CosineDPMSolverMultistepScheduler` (SDE DPM-Solver++, first
or second order, exponential or Karras sigmas, v- or epsilon-prediction, EDM
preconditioning, final sigma zero), checked against diffusers 0.39.0 in
`tests/cpp/test_sampling.cpp`. Its reference implementation draws the SDE noise from a
torchsde Brownian tree; this family draws all noise from its own counter-based
generator instead (`runtime/noise.h`: SplitMix64 words, Box-Muller in double, stream 0
for the initial latents and stream `i + 1` for step `i`). Generated audio therefore
differs from a diffusers run with the same seed, while it is distributed the same way.
`tests/family_noise.py` mirrors the generator so that tests can hand the same noise to
the reference pipeline.

## Tokenizer

The runtime tokenizes with a copy of the repository's T5 unigram tokenizer over the
checkpoint's `tokenizer.json`:
whitespace split, Metaspace pieces, Viterbi over the vocabulary with a run of unknown
characters fused into one `<unk>` (as the tokenizers Unigram model does), truncation to
127 pieces plus `</s>`, padding to 128. transformers 5 builds the T5 tokenizer without a
normalizer, while the runtime applies the part of the file's Precompiled normalizer that
drops control characters and maps Unicode spaces to spaces, so a prompt that contains
control characters can tokenize differently. Special-token text typed into a prompt (for
example `</s>`) is tokenized as ordinary text, while transformers maps it to the special
token. The E2E checks the token ids of every
prompt against the official tokenizer, and `tests/cpp/test_tokenizer.cpp` checks
whitespace, unknown-character and non-ASCII prompts against ids from the tokenizers
library.

## Validation

`tests/test_e2e.py` builds the bundle, runs the native CLI and the C / C++ SDK
consumers, and runs the official diffusers pipeline with the native initial latents and
step noise replayed into it. Each case checks the prompt token ids against the official
tokenizer and these numeric criteria (relative L2):

- waveform: FP32 bundles within 1e-2 of the official FP32 audio (the official pipeline
  itself moves by about 1e-3 when only its attention kernel changes); FP16 bundles no
  further from it than the official FP16 run with the same noise, and never beyond 0.5;
- teacher forced, each engine of the bundle on the recorded inputs of the official FP32
  run: the conditioning engine on every encoded prompt within 1e-4; the denoiser on
  every denoiser call, FP32 within 1e-4 per call, FP16 within 1.25 times the official
  FP16 denoiser's error on the same inputs; the decoder on the final latents, FP32
  within 1e-4, FP16 within 1.25 times the official FP16 decoder's error. A whole
  trajectory amplifies precision noise about a hundredfold (guidance multiplies the
  branch difference every step), so these are the sharp criteria.

| Manifest | Checkpoint | Precision |
| --- | --- | --- |
| `stable-audio-open-1.0.json` | the published checkpoint | FP16 (premerge: `stable-audio-open-1.0-hammer`) |
| `stable-audio-open-1.0-fp32.json` | the published checkpoint | FP32 |
| `stable-audio-random-full*.json` | random weights of the published architecture with the pretrained T5-base encoder, generated by `tests/random_checkpoint.py` | FP16, FP32 |
| `stable-audio-random-tiny.json` | a tiny random checkpoint | FP32 |

`tests/test_engines.py` compares each engine with its diffusers module on the tiny
checkpoint (GPU only; its text projection is a linear layer, the published one is the
identity). On the CPU, `tests/cpp/test_sampling.cpp` checks the noise generator and the
scheduler, `tests/cpp/test_request.cpp` the config fields, checks, sample window and
token framing, and `tests/cpp/test_tokenizer.cpp` the tokenizer.
