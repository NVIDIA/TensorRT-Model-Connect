# TASK-01: Migrate WhisperPipeline + BarkPipeline to TrtModule

## Branch: `agent-X-migrate-whisper-bark`

## Goal

Replace `WhisperBackend` and `BarkBackend` delegation with direct `TrtModule::forward()` calls inside the pipeline classes. After this task, `whisper_backend.cpp/h` and `bark_backend.cpp/h` can be deleted.

## Current State

```
WhisperPipeline → WhisperBackend → DecoderStepEngine + DeviceKvCache + CudaBuffer
BarkPipeline    → BarkBackend    → 4x DecoderStepEngine + 2x DeviceKvCache + embeddings
```

## Target State

```
WhisperPipeline → TrtModule(encoder) + TrtModule(decoder) + KvCache (cross-attn)
BarkPipeline    → TrtModule(semantic) + TrtModule(coarse) + TrtModule(fine) + TrtModule(codec)
                  + KvCache(semantic) + KvCache(coarse) + embedding buffers
```

## WhisperPipeline Migration

**Old backend**: `src/runtime/trt/audio/whisper_backend.cpp` (~400 lines)
**Engines**: 2 (encoder + decoder)

### Engine I/O (from old backend):

**Encoder**:
- Input: `mel_input` [1, n_mels, n_frames] float32
- Output: `encoder_output` [1, seq_len, hidden] float32

**Decoder**:
- Input: `token_id` int32, `position_id` int32, `attention_mask` float32
- Input: `cache_k_N` / `cache_v_N` (self-attention KV cache per layer)
- Input: `cross_k_N` / `cross_v_N` (cross-attention from encoder — bound once)
- Output: `logits` float32, `present_k_N` / `present_v_N`

### Steps:
1. In `WhisperPipeline::transcribe()`:
   - Mel spectrogram extraction (already in pipeline)
   - `encoder_module->forward({{"mel_input", mel_tensor}})` → get encoder output
   - Extract cross-K/V from encoder output, bind to decoder via `decoder_module->bind_external()`
   - Decoder loop: `decoder_module->forward({token_id, position_id, mask})` → argmax → advance KvCache
   - Decode token IDs with tokenizer
2. In `pipeline_factory.cpp`: create 2 TrtModules + KvCache instead of calling `make_whisper_pipeline_from_bundle()`
3. Delete `audio_backend_factory.cpp` whisper section

## BarkPipeline Migration

**Old backend**: `src/runtime/trt/audio/bark_backend.cpp` (~600 lines)
**Engines**: 4 (semantic + coarse + fine + codec)

### Steps:
1. Port the 3-stage generation: semantic → coarse → fine
2. Each stage is a decoder loop with KvCache
3. Embedding lookups from bundle sections (semantic_embed, coarse_embed, fine_embed)
4. Codec decoder: single forward pass, no KV cache
5. Waveform assembly from codec output

## Files to Modify
- `src/runtime/pipelines/audio_pipeline.h` — update WhisperPipeline/BarkPipeline members
- `src/runtime/pipelines/audio_pipeline.cpp` — port inference loops
- `src/runtime/pipeline_factory.cpp` — create TrtModules directly
- `src/runtime/pipelines/audio_backend_factory.cpp` — remove whisper/bark sections

## Files to Delete (after verification)
- `src/runtime/trt/audio/whisper_backend.cpp/h`
- `src/runtime/trt/audio/bark_backend.cpp/h`

## Verification
```bash
pytest tests/test_e2e.py::test_e2e[whisper-tiny] tests/test_e2e.py::test_e2e[whisper-large-v3-turbo] tests/test_e2e.py::test_e2e[canary-1b-v2] tests/test_e2e.py::test_e2e[bark-small] tests/test_e2e.py::test_e2e[bark-large] -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python
```

All 5 models must pass E2E. Output must be identical to pre-migration.
