# VibeVoice text-to-speech

This family builds `microsoft/VibeVoice-1.5B` (`VibeVoiceForConditionalGeneration`, the original Microsoft
checkpoint layout) for the `text_to_speech` Task. It synthesizes 24 kHz mono speech for one or more speakers,
optionally cloning each speaker's voice from a reference recording. The converted Transformers layout
(`vibevoice/VibeVoice-1.5B-hf`) uses different weight and config names and is not claimed.

## What runs where

The checkpoint holds four towers: a Qwen2.5-1.5B decoder, a sigma-VAE acoustic tokenizer (encoder and decoder,
3200x downsampling to 7.5 frames per second), a semantic tokenizer encoder, and a diffusion head that predicts
one 64-dimensional acoustic latent per frame. The bundle has five engines:

| Engine | Graph |
|---|---|
| `lm.plan` | Qwen2 decoder, prefill and decode profiles over a runtime-owned KV cache; returns the last hidden row, the four control-token logits and the vocabulary log-normalizer |
| `embed.plan` | token embedding gather |
| `diffusion_head.plan` | one v-prediction for the conditional and unconditional LM states |
| `feedback.plan` | one frame: latent -> acoustic decoder (3200 samples) -> semantic encoder -> next LM input embedding; every causal convolution carries its state across frames in one flat tensor |
| `voice_encoder.plan` | voice-prompt audio -> acoustic latents (with the sigma-VAE sampling noise) -> LM input embeddings |

The runtime tokenizes the prompt (Qwen2.5 byte-level BPE), runs the generation loop and the multistep
DPM-Solver++ on the host, and returns float32 PCM. Generation follows Transformers' `VibeVoiceForConditionalGeneration.generate`
(5.22.2) with its released generation config: control tokens chosen greedily among `<|endoftext|>`,
`<|vision_start|>`, `<|vision_end|>` and `<|vision_pad|>`; classifier-free guidance 1.3 on the diffusion head and,
as Transformers also applies it because the guidance scale is set, on the control-token log-probabilities
(a third LM sequence of raw token ids); 10 DPM-Solver++ steps (`squaredcos_cap_v2`, v-prediction).

## Input text

`text` is either plain text, spoken by speaker 0, or a script whose every non-empty line reads
`Speaker <id>: <text>`. Ids written from 1 (`Speaker 1`, `Speaker 2`) are renumbered from 0, as the model's
original script parser does. A script mixing speaker lines and other lines is refused, as is text containing a
tokenizer special token. The model speaks English and Chinese and has no language control: a request that
states a `language` is refused.

## Configuration

| Field | Type | Default | Meaning |
|---|---|---|---|
| `seed` | int | 0 | seed of the diffusion noise and the voice-prompt sampling noise |
| `max_new_tokens` | int | until the model stops or the cache is full | most generated tokens (speech frames and control tokens) |
| `cfg_scale` | float | 1.3 | classifier-free guidance scale (diffusion head and control tokens) |
| `diffusion_steps` | int | 10 | DPM-Solver++ steps per frame |
| `voice_prompts` | list of paths | empty | 24 kHz mono WAV files (16/24/32-bit PCM or 32-bit float), the voice of speaker 0, 1, ... in order; at most 60 s each |

```sh
trtmc generate-audio vibevoice.bundle --prompt $'Speaker 1: Hello there.\nSpeaker 2: Hi!' \
    --output out.wav --seed 7 --set 'voice_prompts=["alice.wav", "frank.wav"]'
```

`max_sequence_length` at build time sets the positions each sequence may hold (prompt plus generated tokens;
default 16384, from 64 to 65536); each extra second of speech takes 7.5 positions. `precision=bf16` (default, the
checkpoint's type) or `precision=fp32`. TensorRT has no bfloat16 deconvolution, so the acoustic decoder's
transposed convolutions run in float32 inside the bfloat16 feedback engine (bfloat16 inputs and weights, float32
sums, one rounding back).

## Noise and reproducibility

The diffusion and voice-prompt noise come from a fully specified stream (`runtime/noise.h`: SplitMix64 and a
double-precision Box-Muller transform) seeded by `seed`, drawn in this order: for each voice prompt one scale
draw and `frames x 64` latent draws (frame-major), then 64 draws per generated frame. The E2E feeds the
Transformers reference the same draws, so both sides synthesize from identical noise. In bfloat16 bundles the
host rounds every value the reference holds in bfloat16 where the reference rounds it.

## Validation

`tests/test_engine_parity.py` compares every engine with the Transformers modules on a small random checkpoint
in the original layout (float32). `tests/cpp/test_host.cpp` checks the host side against
`tests/data/host_fixtures.json` (diffusers' scheduler in float32 and in bfloat16 on CUDA, the noise stream, script
parsing, prompt rendering; with `VIBEVOICE_TOKENIZER_JSON` also the token ids of the reference tokenizer);
`tests/test_host_fixtures.py` keeps the fixtures current. `tests/test_e2e.py` builds the bundle, synthesizes with
the native CLI, runs the reference with the injected noise and transcribes the native audio with
`openai/whisper-large-v3-turbo`: float32 cases must match the reference frame count and each of the first 12
frames (60 dB or more), bfloat16 cases the early frames and length of the official bfloat16 synthesis, and every
case must say its text; the C and C++ SDK consumers (`tests/sdk_consumer.c`, `tests/sdk_consumer.cpp`) then
synthesize from the same bundle. The voice-cloning case clones a voice the family synthesized itself, so no
third-party recording is needed. The float32 long-prompt case (582 prompt tokens, two prefill chunks) stops at
`max_new_tokens` before its script ends, so its transcript is compared with the reference's.

## Responsible use

The model card limits VibeVoice to research use and lists out-of-scope uses (voice impersonation without
consent, disinformation, live voice conversion). It also states that Microsoft embeds an audible disclaimer and
an imperceptible watermark in synthesized audio; neither is part of the checkpoint or of the Transformers
generation this family follows, and the family returns the model's audio unchanged.
