# TASK-04: Delete Legacy Backends + Factory Files + Shared Infrastructure

## Branch: `agent-X-cleanup-legacy`

## Goal

After TASK-01, TASK-02, and TASK-03 are complete, delete all old backend code,
factory adapters, and shared legacy infrastructure (`DecoderStepEngine`,
`DeviceKvCache`, `DeviceResources`, etc.).

## Blocked On

- TASK-01 (Whisper + Bark migrated)
- TASK-02 (Magpie + Speech + Omni migrated)
- TASK-03 (Flux + Wan + ZImage migrated)

## Files to Delete

### Old audio backends
- `src/runtime/trt/audio/whisper_backend.cpp/h`
- `src/runtime/trt/audio/bark_backend.cpp/h`
- `src/runtime/trt/audio/magpie_tts_backend.cpp/h` + helper files
- `src/runtime/trt/audio/speech_backend.cpp/h` + helper files
- `src/runtime/trt/audio/omni_backend.cpp/h`

### Old diffusion backends
- `src/runtime/trt/diffusion/flux_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/wan_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/z_image_diffusion_backend.cpp/h`
- `src/runtime/trt/diffusion/diffusion_backend_base.cpp/h`
- `src/runtime/trt/diffusion/diffusion_backend.h` (IDiffusionBackend interface)

### Factory adapters (bridge between new pipelines and old backends)
- `src/runtime/pipelines/audio_backend_factory.cpp/h`
- `src/runtime/pipelines/diffusion_backend_factory.cpp/h`

### Legacy shared infrastructure
- `src/runtime/trt/core/trt_engine_lifecycle.cpp/h` (DecoderStepEngine)
- `src/runtime/trt/core/device_kv_cache.cpp/h` (DeviceKvCache, DeviceResources)
- `src/runtime/trt/core/trt_decode_runtime.cpp/h` (select_argmax_token, build_attention_mask)
- `src/cabi/bundle/bundle_helpers.cpp/h` — remove `make_decoder_engine()` and related helpers (keep `find_bundle_sections()`, `extract_tokenizer_from_bundle()`, `load_mel_filterbank()`)

### Old VL/perception backends (if not already deleted)
- `src/runtime/trt/multimodal/vl_backend.cpp/h` (replaced by VLPipeline)
- `src/runtime/trt/perception/segmentation_backend.cpp/h` (replaced by SegmentPipeline)

## Steps

1. Verify no remaining `#include` references to deleted files
2. Remove from `CMakeLists.txt` `trtmc_core` source list
3. Remove associated test files that only test old backends
4. Update `CLAUDE.md` source layout section
5. Run full build + ctest + E2E smoke test

## Verification
```bash
# Build must succeed without old files
cmake --build build -j

# All unit tests pass
ctest --test-dir build --output-on-failure

# E2E smoke test across modalities
pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] tests/test_e2e.py::test_e2e[bert-base-uncased] tests/test_e2e.py::test_e2e[qwen25vl-3b] tests/test_e2e.py::test_e2e[whisper-tiny] tests/test_e2e.py::test_e2e[bark-small] tests/test_e2e.py::test_e2e[flux-schnell] tests/test_e2e.py::test_e2e[segformer-b0-ade] tests/test_e2e.py::test_e2e[mamba-130m] -v --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python

# CCN gate
python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

## Expected Impact

- ~5000-8000 lines of old backend code deleted
- ~1000 lines of factory adapter code deleted
- ~500 lines of legacy infrastructure deleted
- Cleaner dependency graph: pipeline → TrtModule → TRT (no intermediate layer)
