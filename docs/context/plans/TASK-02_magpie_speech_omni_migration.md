# TASK-02: Migrate MagpiePipeline + SpeechPipeline + OmniPipeline to TrtModule

## Branch: `agent-X-migrate-magpie-speech-omni`

## Goal

Replace `MagpieTTSBackend`, `SpeechToSpeechBackend`, and `OmniBackend` delegation with direct `TrtModule::forward()` calls. These are the most complex audio pipelines.

## Current State

```
MagpiePipeline  → MagpieTTSBackend     → 3 engines + KvCache + CFG dual-cache + IPA tokenizer
SpeechPipeline  → SpeechToSpeechBackend → 6+ engines + KvCache + depth state + Mimi codec
OmniPipeline    → OmniBackend           → 3 engines + KvCache
```

## MagpiePipeline Migration (~500 LOC)

**Old backend**: `src/runtime/trt/audio/magpie_tts_backend.cpp` (~1600 lines)
**Engines**: 3 (text_encoder + decoder + codec)

### Key complexity:
- CFG (classifier-free guidance): runs decoder twice per step (conditional + unconditional)
- Dual KvCache: one for conditional, one for unconditional path
- Context frame prefill: 110 pre-computed context frames
- IPA tokenizer (not HF tokenizer)
- Embedding lookups: text_embed, audio_embed, context_embed, context_lengths
- Repetition detection and text-completion stopping heuristic
- GPU greedy loop with CUDA kernels (optional fast path)

### Steps:
1. Port text encoder: `encoder_module->forward({input_ids})` → cross-attention embedding
2. Port decoder loop with CFG: two `decoder_module->forward()` per step
3. Port codec: `codec_module->forward({codes})` → waveform
4. Handle embedding lookups as host-side operations before forward()
5. Port the `set_decoder_vocab_from_logits()` pattern for correct logits allocation

## SpeechPipeline Migration (~800 LOC) — Most Complex

**Old backend**: `src/runtime/trt/audio/speech_backend.cpp` (~1800 lines)
**Engines**: 6+ (temporal_decoder + 6x depth_decoder + mimi_encoder + mimi_decoder)

### Key complexity:
- Mimi encoder: waveform → codec tokens
- Temporal decoder: autoregressive with KvCache, text prompt injection
- Depth decoders: 6 separate engines for codebook refinement
- Delay pattern: interleaved temporal+depth execution
- Embedding lookups: audio_embeddings, temporal_text, depth_text, depth_audio, depth_projection
- Mimi decoder: codec tokens → waveform (separate TRT engine with fixed input shape)
- Tail frames: continue generating beyond input duration

### Steps:
1. Port Mimi encoder as TrtModule
2. Port temporal decoder with KvCache + embedding injection
3. Port depth decoders (6 TrtModules, each with its own mini-KvCache)
4. Port delay pattern interleaving logic
5. Port Mimi decoder as TrtModule
6. Handle the stochastic sampling (temperature + top_k)

## OmniPipeline Migration (~400 LOC)

**Old backend**: `src/runtime/trt/audio/omni_backend.cpp` (~600 lines)
**Engines**: 3 (text_encoder + talker + code2wav)

### Steps:
1. Port text encoder: `encoder->forward({input_ids})`
2. Port talker decoder loop with KvCache
3. Port code2wav: `codec->forward({codes})` → waveform

## Files to Modify
- `src/runtime/pipelines/audio_pipeline.h/cpp`
- `src/runtime/pipeline_factory.cpp`
- `src/runtime/pipelines/audio_backend_factory.cpp` — remove magpie/speech/omni sections

## Files to Delete (after verification)
- `src/runtime/trt/audio/magpie_tts_backend.cpp/h` + helper files
- `src/runtime/trt/audio/speech_backend.cpp/h` + helper files
- `src/runtime/trt/audio/omni_backend.cpp/h`
- `src/runtime/pipelines/audio_backend_factory.cpp/h` (once all audio migrated)

## Verification
```bash
pytest tests/test_e2e.py::test_e2e[magpie-tts-357m] tests/test_e2e.py::test_e2e[personaplex-7b] -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

## Depends On
- TASK-01 (establishes the audio migration pattern with simpler Whisper/Bark)
