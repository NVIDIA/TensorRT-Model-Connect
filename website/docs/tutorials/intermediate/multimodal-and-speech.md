---
title: Intermediate Tutorial - Multimodal and Speech
---

import Diagram from '@site/src/components/Diagram';

This tutorial exercises non-text modalities that still use the same bundle/runtime contract.

## Learning objectives

By the end of this lab, you should be able to compare vision-language, offline
ASR, streaming ASR, and text-to-audio pipelines by preprocessing, engine
sections, iterative state, public task method, and result type.

Select the CLI before running an example:

```bash
export TRTMC=trtmc
# Source build inside the development container:
# export TRTMC=./build/trtmc
```

Commands that use `tests/...` sample assets require a repository checkout and
must run from its root; the CLI itself may still come from the installed wheel.

By this point you should recognize the common pattern:

<Diagram
  src="/img/diagrams/tutorials/intermediate/task-pipeline-patterns.svg"
  alt="Shared task pipeline from typed input through task-owned preprocessing, TensorRT engines, postprocessing, and a typed result"
  caption="The public pipeline shape stays stable while each model owner supplies the preprocessing, engine topology, processing loop, and typed result for its task."
/>

The difference between text, vision, and speech is mostly the preprocessing, component layout, and postprocessing. The bundle and plugin registry model stays the same.

## Vision-language

```bash
$TRTMC build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl.bundle \
  --precision fp16 \
  --max-cache-length 384

$TRTMC run /tmp/qwen25vl.bundle \
  --prompt "Describe this image in one sentence." \
  --image tests/assets/test_image.jpg \
  --max-new-tokens 48
```

The family plugin builds a vision encoder and a text decoder. The Qwen-VL
runtime plugin creates `QwenVlPipeline`, preprocesses the image, injects image
embeddings into the prompt flow, and decodes text. InternVL uses its own
`InternVlPipeline` in a separate model DSO.

On GitHub `main` commit `e6b798cdb145c38caf1ede8eda7f5ce83f894138`,
Qwen2.5-VL opts its embed-input decoder into separate `prefill_engine_plan`
and `engine_plan` sections; the runtime loads both and uses the matching
prefill/decode module. That split applies to the default single-GPU,
fixed-cache, non-TriAttention request. Qwen3-VL's deepstack decoder,
tensor-parallel builds, dynamic-KV/TriAttention builds, and an explicit
`dual_profile` request use their supported fallback layouts. Inspect
`config.json` and the section list instead of assuming every Qwen-VL
generation uses the same engine shape.

The split decoder can opt into decomposed decode attention with
`--set qwen_vl_decoder.decode_attention=decomposed`. The same namespace exposes
`max_prefill_length`, `opt_prefill_length`, and `builder_workspace_gib` as
build-time profile controls. Decomposed attention is rejected outside an active
split-decode build. For dynamic Qwen2.5-VL images, the runtime derives the
merged mRoPE grid from the preprocessed image grid instead of assuming the
fixed-profile dimensions.

The Qwen3-VL decoder builder also accepts BF16. In that mode the decoder and
KV cache use BF16 while the separate vision tower stays FP32:

```bash
$TRTMC build Qwen/Qwen3-VL-2B-Instruct \
  -o /tmp/qwen3vl-bf16.bundle \
  --precision bf16 \
  --max-cache-length 384
```

This is an implementation contract, not a qualification claim: the checked-in
`qwen3-vl-2b` E2E manifest still selects FP16. Retain target-hardware parity
evidence before presenting a BF16 build as qualified.

Qwen2.5-VL can instead build a dynamic vision profile that applies Qwen
smart-resize at runtime:

```bash
$TRTMC build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl-dynamic.bundle \
  --precision fp16 \
  --max-cache-length 384 \
  --set qwen_vl_vision.dynamic_resolution=true
```

This mode preserves the source aspect ratio, aligns the resized dimensions to
the model's patch/merge factor, and uses the checkpoint's packaged pixel limits
when available. Qwen3-VL does not currently support the dynamic profile. Use
the fixed-profile command above for that family.

<Diagram
  src="/img/diagrams/tutorials/intermediate/vision-language-pipeline.svg"
  alt="Vision-language pipeline merging tokenized text and image embeddings before following the decoder layout recorded in the bundle"
  caption="Text and vision branches merge before generation; inspect the bundle to determine whether the decoder uses a combined fallback layout or the qualified split prefill and decode plans."
/>

Key ideas:

| Concept | Meaning |
| --- | --- |
| Image preprocessing | Resize, crop/pad, normalize, and lay out pixels as the vision encoder expects. |
| Image embeddings | Numeric representation of the image produced by the vision component. |
| Prompt injection | The runtime inserts image embeddings into the text decoder flow at model-specific placeholder positions. |
| Output | Text generation still uses the decoder loop and sampler. |

## Speech-to-text

```bash
$TRTMC build openai/whisper-large-v3-turbo \
  -o /tmp/whisper.bundle \
  --precision fp16

$TRTMC transcribe /tmp/whisper.bundle \
  --audio tests/e2e/models/whisper/data/Recording.wav \
  --max-new-tokens 224
```

The Whisper bundle emits the model-owned
`whisper_speech_to_text` strategy. Its runtime uses audio preprocessing, mel
feature extraction, encoder/decoder execution, and text decoding. Canary uses
the separate `canary_speech_to_text` strategy for the same broad task.

For local Canary checkpoints, multilingual prompts, beam search, batching, and
segment controls, continue with [Configurable Canary Decoding](canary-decoding.md).

<Diagram
  src="/img/diagrams/tutorials/intermediate/speech-to-text-pipeline.svg"
  alt="Speech-to-text pipeline converting PCM audio into mel features before encoder, decoder, and transcript generation"
  caption="Speech runtimes prepare feature frames before model execution; raw waveform is not passed directly to the text decoder."
/>

The important beginner mistake is to treat audio as if it were text. Speech models usually do not consume raw waveform directly inside the decoder. They first convert audio into feature frames.

## Streaming ASR

```bash
$TRTMC build nvidia/nemotron-speech-streaming-en-0.6b \
  -o /tmp/nemotron-rnnt.bundle \
  --precision fp16 \
  --max-cache-length 128

$TRTMC transcribe /tmp/nemotron-rnnt.bundle \
  --audio tests/e2e/models/whisper/data/Recording.wav \
  --stream \
  --chunk-ms 160 \
  --att-context-size 70,13
```

Cache-aware streaming uses `TranscriptionStreamConfig` in `include/trtmc/pipeline.h`. Right contexts `{0, 1, 6, 13}` correspond to the supported FastConformer-RNNT schedules documented in code comments.

<Diagram
  src="/img/diagrams/tutorials/intermediate/streaming-asr-sequence.svg"
  alt="Streaming ASR sequence showing early chunks buffered until the schedule is ready, a later partial result, and finish processing all pending features"
  caption="accept_audio() always returns the current TranscriptionStreamResult, which may still be empty before enough features are ready. finish() runs process_ready(true) so pending features and remaining RNNT frames are processed before the final result."
  sequence
/>

Streaming adds two concerns that offline transcription does not have:

| Concern | Why it matters |
| --- | --- |
| Chunk schedule | The model only sees part of the audio at a time. |
| Right context | Some future frames are needed for accuracy; larger right context increases latency. |
| Feature cache | Overlapping audio/features should not be recomputed every chunk. |
| Partial results | Applications may display interim hypotheses before finalization. |

## Text-to-audio

```bash
$TRTMC build nvidia/magpie_tts_multilingual_357m \
  -o /tmp/magpie.bundle \
  --precision fp16

$TRTMC generate-audio /tmp/magpie.bundle \
  --prompt "A calm narration for a product demo." \
  --output /tmp/out.wav
```

Magpie supports chunked audio callbacks in the C++ API and `trtmc serve-audio` in the CLI.

<Diagram
  src="/img/diagrams/tutorials/intermediate/text-to-audio-pipeline.svg"
  alt="Text-to-audio pipeline from prompt and phoneme tokens through acoustic and codec stages to PCM output"
  caption="The task returns audio samples as an AudioResult, a WAV file, or streaming chunks rather than returning generated text."
/>

Text-to-audio is a good example of why `IPipeline` has task-specific methods. The output is not token IDs or a string; it is `AudioResult` or chunks delivered through `generate_audio_streaming`.

## What to inspect in multimodal bundles

| Modality | Inspect for |
| --- | --- |
| Vision-language | Vision engine section, tokenizer assets, image preprocessing config, and an owner-qualified key such as `qwen_vl_vision_language` or `internvl_vision_language`. |
| Speech-to-text | Audio feature metadata, tokenizer assets, encoder/decoder sections, and `whisper_speech_to_text` or `canary_speech_to_text` for those families. |
| Streaming ASR | RNNT/streaming config, supported context schedule, and `nemotron_speech_streaming_speech_to_text_rnnt`. |
| Text-to-audio | Acoustic and codec sections, tokenizer/phoneme assets, audio sample-rate metadata, and `text_to_audio_magpie` for Magpie. |

## Self-check

1. Why can two tasks use the same `.bundle` container without sharing a runtime
   strategy?
2. Which state makes streaming ASR different from offline transcription?
3. Why should a text-to-audio result not be handled like generated token IDs?

<details>
<summary>Check your answers</summary>

1. The bundle is a common artifact boundary, while each model owns its
   preprocessing, engine topology, task loop, strategy, and typed result.
2. Streaming retains chunk schedule, feature/encoder cache, right-context, and
   partial-hypothesis state across calls.
3. The public contract returns audio samples or chunks with sample-rate/output
   metadata, not a text-decoder token sequence.

</details>

{/* Collaborative review anchor. */}
