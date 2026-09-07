---
title: Intermediate Tutorial - Multimodal and Speech
---

import Diagram from '@site/src/components/Diagram';

## Learning objectives

- run vision-language, transcription, streaming ASR, and audio-generation Tasks;
- locate every model's dependencies, assets, runtime, and tests in one family;
- understand that shared Task interfaces do not create shared model pipelines.

<Diagram
  src="/img/diagrams/tutorials/intermediate/task-pipeline-patterns.svg"
  alt="Typed multimodal input passes through family-owned preprocessing and TensorRT engines before a typed Task result"
  caption="Each family implements the abstract user behavior with its own processing loop."
/>

Use one installed runtime root for the examples:

```bash
export TRTMC_RUNTIME_ROOT=/opt/trtmc/lib
```

## Vision-language

Build the exact Qwen2.5-VL example declared by its family:

```bash
python -m pip install -r families/qwen_vl/requirements.txt
python -m tensorrt_model_connect build Qwen/Qwen2.5-VL-3B-Instruct \
  --output qwen25vl-3b.bundle \
  --precision fp32 \
  --max-sequence-length 384
```

Run the public vision-language Task:

```bash
trtmc run qwen25vl-3b.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "What color is the vehicle in this image? Answer in one word." \
  --image families/qwen_vl/tests/data/test_img.jpeg \
  --max-new-tokens 10
```

<Diagram
  src="/img/diagrams/tutorials/intermediate/vision-language-pipeline.svg"
  alt="Text and image inputs are prepared by the Qwen-VL family and meet in its model-owned decoder before a TextResult is returned"
  caption="Qwen-VL owns image preparation, engine layout, tokenization, and decoding behind IVisionLanguageGeneration."
/>

Do not infer Qwen-VL's internal engine layout from another family. Inspect the
bundle sections and `families/qwen_vl/model.py` for this exact build.

## Speech-to-text

```bash
python -m pip install -r families/whisper/requirements.txt
python -m tensorrt_model_connect build openai/whisper-large-v3-turbo \
  --output whisper-large-v3-turbo.bundle \
  --precision fp32 \
  --max-sequence-length 128

trtmc transcribe whisper-large-v3-turbo.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input families/whisper/tests/data/Recording.wav \
  --max-output-tokens 20
```

<Diagram
  src="/img/diagrams/tutorials/intermediate/speech-to-text-pipeline.svg"
  alt="PCM audio becomes model-owned features, encoder and decoder outputs, and a transcript"
  caption="Whisper implements ITranscription; its feature extraction and decoder stay in the Whisper family."
/>

The CLI owns WAV decoding. `ITranscription` accepts samples and a typed request,
so applications do not depend on CLI file helpers.

## Streaming ASR

Use the dedicated streaming family and Task identity:

```bash
python -m pip install -r families/nemotron_speech_streaming/requirements.txt
python -m tensorrt_model_connect build nvidia/nemotron-speech-streaming-en-0.6b \
  --output nemotron-streaming.bundle \
  --precision fp16 \
  --max-sequence-length 128

trtmc transcribe-streaming nemotron-streaming.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --input families/nemotron_speech_streaming/tests/data/Recording.wav \
  --chunk-samples 17920
```

<Diagram
  src="/img/diagrams/tutorials/intermediate/streaming-asr-sequence.svg"
  alt="An application feeds audio chunks into a family-owned transcription stream and receives partial then final results"
  caption="The abstract stream contract is shared; cache scheduling and RNNT execution are family-owned."
  sequence
/>

Chunk size, attention context, buffering, and finalization are Task-request or
family concerns. Another transcription family does not become streaming-capable
because this CLI command exists.

## Text-to-audio

```bash
python -m pip install -r families/magpie_tts/requirements.txt
python -m tensorrt_model_connect build nvidia/magpie_tts_multilingual_357m \
  --output magpie-tts-357m.bundle \
  --precision fp32 \
  --max-sequence-length 512

trtmc generate-audio magpie-tts-357m.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "Please say this sentence clearly." \
  --output magpie.wav \
  --max-new-tokens 260 \
  --seed 42
```

<Diagram
  src="/img/diagrams/tutorials/intermediate/text-to-audio-pipeline.svg"
  alt="Text becomes family-owned tokens, acoustic or codec representations, and PCM audio"
  caption="Magpie owns its text processing, engines, codec path, and audio evidence behind IAudioGeneration."
/>

The family dependency file includes packages used by its build and official
reference. Those packages are not added to the shared base environment.

## What to inspect in multimodal bundles

```bash
trtmc inspect BUNDLE
```

Check `family`, `task`, `backend`, and section names. The format-1 core does not
standardize image, audio, tokenizer, or component-plan section names; each
family writes and reads its own.

For exact evidence, use the matching directories:

```text
families/qwen_vl/tests/
families/whisper/tests/
families/nemotron_speech_streaming/tests/
families/magpie_tts/tests/
```

## Self-check

1. Why does the Task API accept pixels or samples rather than file paths?
2. Which family owns streaming cache behavior?
3. Why can two audio families duplicate helpers without importing each other?
4. What does bundle inspection prove, and what still requires E2E evidence?
