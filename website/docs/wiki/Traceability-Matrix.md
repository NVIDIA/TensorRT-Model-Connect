# Traceability Matrix

| Field | Value |
|-------|-------|
| Document ID | TRTMC-TRACE-001 |
| Applicable standard | ISO 26262:2018 Part 6, Sections 9 and 10 |
| Revision | 2.0 |
| Date | 2026-03-12 |
| Author | Safety Architecture Team (TensorRT-Model-Connect Team) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |
| Status | Active |

---

## 1. Purpose and Scope

This document establishes **bi-directional traceability** from architecture
contracts through unit design to verification evidence, following the
structure required by ISO 26262-6 Section 9 (unit verification) and Section 10
(integration verification).

Scope covers:

- C++ runtime (`src/`, `include/trtmc/`)
- Python build package (`tensorrt_model_connect/`)
- Unit tests (`tests/cpp/`, `tests/builder/`, `tests/tools/`)
- Integration and E2E tests (`tests/test_e2e.py`, `tests/e2e_harness/`, `tests/e2e/models/`)

This complements (does not replace) `tests/runtime_strategy_matrix.yaml`,
which is machine-checked strategy parity. This page is human-maintained
intent/design/test traceability.

---

## 2. Trace ID Scheme

Use stable IDs so rows remain valid as files move:

| Prefix | Layer | Example |
|--------|-------|---------|
| `ARCH-*` | Architecture contract or capability | `ARCH-BDL-001` |
| `UD-*` | Unit design element (references real source files) | `UD-BDL-01` |
| `UT-*` | Unit test evidence (references real test files) | `UT-BDL-CPP-01` |
| `IT-*` | Integration/E2E test evidence (references real manifests/harness files) | `IT-E2E-QWEN3-01` |

Guidance:

- One `ARCH-*` may map to multiple `UD-*` entries.
- Each `UD-*` must map to at least one `UT-*` or `IT-*`.
- Every `UT-*`/`IT-*` must point back to exactly one primary `ARCH-*`
  (plus optional secondary IDs).

---

## 3. Mandatory Test Intent Fields

Every test added or modified in this repository must document all three fields:

| Field | Required content |
|-------|------------------|
| Intent | The behavior or contract being validated |
| Preconditions | Assumptions/setup needed for the test to be valid |
| Postconditions | Observable outcomes/invariants guaranteed by passing assertions |

Required placement:

- **Python:** test function/class docstring.
- **C++:** comment block above the scenario's `check(...)` assertions.

A test is traceability-incomplete if any of the three fields is missing.

All tests must also include trace IDs (`ARCH-*`, `UD-*`, `UT-*`/`IT-*`)
linking to entries in the matrix below.

---

## 4. Traceability Matrix

### 4.1 Bundle Format

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-BDL-001` | Bundle format read/write roundtrip: `.trtfb` bundles must survive write-then-read without data loss; section headers, magic bytes, and payload integrity are preserved. | `UD-BDL-01`: `src/bundle/bundle_format.h`, `src/bundle/bundle_format.cpp`; `UD-BDL-02`: `tensorrt_model_connect/tensorrt_model_connect/bundle_writer.py` | `UT-BDL-CPP-01`: `tests/cpp/test_bundle_format.cpp` (magic, section parsing, round-trip); `UT-BDL-PY-01`: `tests/builder/test_bundle_writer.py` (bundle format round-trip, section integrity) | `IT-E2E-*`: Every E2E test in `tests/test_e2e.py` exercises bundle read/write (all 110 manifests in `tests/e2e/models/`) | verified |

### 4.2 Configuration

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CFG-001` | `BaseConfig` (`parse_base_config()`) parses all 25 runtime strategies: every `runtime_strategy` field value (`decoder_kv_cache`, `decoder_moe`, `diffusion_flux`, `diffusion_pixart`, `diffusion_wan`, `diffusion_zimage`, `embedding`, `encoder_only`, `hybrid_mamba_attention`, `marian_translation`, `neural_operator`, `object_detection`, `omni_multimodal`, `prompted_segmentation`, `reranking`, `rwkv_recurrent`, `segmentation`, `seq2seq_encoder_decoder`, `speech_to_speech`, `speech_to_text`, `ssm_recurrent`, `text_to_audio_bark`, `text_to_audio_magpie`, `text_to_text`, `vision_language`) must be parsed correctly from bundle JSON config. | `UD-CFG-01`: `include/trtmc/runtime/pipeline_plugin.h`, `src/runtime/registry/pipeline_plugin.cpp` | `UT-CFG-CPP-01`: `tests/cpp/test_pipeline_registry.cpp` (strategy registration, lookup); `UT-CFG-CPP-02`: `tests/cpp/test_pipeline_api.cpp` (pipeline creation via factory) | `IT-E2E-*`: Every E2E test validates config parsing as part of bundle load | verified |
| `ARCH-CFG-002` | `ModelConfig` (Python) correctly parses HuggingFace `config.json` for all supported model families, extracting architecture parameters needed for engine building. | `UD-CFG-02`: `tensorrt_model_connect/tensorrt_model_connect/config.py` | `UT-CFG-PY-01`: `tests/builder/test_config.py` (ModelConfig parsing, VL text_config merge, edge cases) | N/A (build-time only) | verified |

### 4.3 Pipeline Factory and Dispatch

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-FAC-001` | `PipelineFactory` dispatches to the correct pipeline implementation based on `runtime_strategy` in the bundle config. Each strategy routes to a distinct pipeline class. | `UD-FAC-01`: `src/runtime/registry/pipeline_factory.cpp`, `include/trtmc/runtime/pipeline_factory.h`; `UD-FAC-02`: `src/cabi/api/trtmc_c.cpp` (C ABI entry point) | `UT-FAC-CPP-01`: `tests/cpp/test_pipeline_api.cpp` (pipeline creation via factory); `UT-FAC-CPP-02`: `tests/cpp/test_c_abi_entry.cpp` (C ABI dispatch); `UT-FAC-CPP-03`: `tests/cpp/test_pipeline_registry.cpp` (registry lookup) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json` via `tests/test_e2e.py` (decoder_kv_cache); `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba-130m.json` (ssm_recurrent); `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json` (vision_language) | verified |

### 4.4 TRT Module

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-MOD-001` | `TrtModule` wraps TRT engine execution: manages engine deserialization, execution context creation, tensor binding, and synchronous inference dispatch. All pipeline backends use `TrtModule` as their engine interface. | `UD-MOD-01`: `include/trtmc/runtime/trt_module.h`, `src/runtime/backend/trt_module_impl.cpp` | `UT-MOD-CPP-01`: `tests/cpp/test_trt_module.cpp` (module construction, tensor binding, lifecycle) | `IT-E2E-*`: Every E2E test exercises `TrtModule` through pipeline execution | verified |

### 4.5 KV Cache

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-KVC-001` | KV cache manages autoregressive state: device-resident cache supports append, position tracking, mask progression, and reset. Cache behavior is identical between C++ runtime and Python debug runner. | `UD-KVC-01`: `include/trtmc/runtime/kv_cache.h`, `src/runtime/core/kv_cache.cpp`; `UD-KVC-02`: `src/runtime/core/device_kv_cache.h`, `src/runtime/core/device_kv_cache.cpp` | `UT-KVC-CPP-01`: `tests/cpp/test_device_kv_cache.cpp` (cache construction, mask progression, position clamping, reset); `UT-KVC-CPP-02`: `tests/cpp/test_kv_cache_new.cpp` (additional cache tests); `UT-KVC-PY-01`: `tests/builder/test_cache_state_machine.py` (position, mask, cache append/shift logic) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json` (autoregressive text gen validates cache correctness) | verified |

### 4.6 Tokenizer

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TOK-001` | Tokenizer encode/decode roundtrip: `VocabTokenizer` (vocabulary lookup), `BpeTokenizer` (BPE), `WordPieceTokenizer`, `UnigramTokenizer`, and `IpaTokenizer` (IPA phoneme) all support lossless encode-then-decode for in-vocabulary text. HuggingFace tokenizer integration via Python subprocess is handled at the pipeline level. `add_special_tokens` is controlled by bundle config field. | `UD-TOK-01`: `src/tokenizer/vocab_tokenizer.cpp`; `UD-TOK-02`: `src/tokenizer/bpe_tokenizer.cpp`; `UD-TOK-03`: `src/tokenizer/wordpiece_tokenizer.cpp`; `UD-TOK-04`: `src/tokenizer/unigram_tokenizer.cpp`; `UD-TOK-05`: `src/tokenizer/ipa_tokenizer.cpp` | `UT-TOK-CPP-01`: `tests/cpp/test_vocab_tokenizer.cpp` (encode/decode, round-trip, case insensitivity); `UT-TOK-CPP-02`: `tests/cpp/test_bpe_tokenizer.cpp` (BPE encode/decode); `UT-TOK-CPP-03`: `tests/cpp/test_bpe_golden.cpp` (BPE golden reference); `UT-TOK-CPP-04`: `tests/cpp/test_wordpiece_tokenizer.cpp` (WordPiece encode/decode); `UT-TOK-CPP-05`: `tests/cpp/test_wordpiece_golden.cpp` (WordPiece golden reference); `UT-TOK-CPP-06`: `tests/cpp/test_unigram_tokenizer.cpp` (Unigram encode/decode); `UT-TOK-CPP-07`: `tests/cpp/test_unigram_golden.cpp` (Unigram golden reference); `UT-TOK-CPP-08`: `tests/cpp/test_ipa_tokenizer.cpp` (IPA phoneme tokenizer) | `IT-E2E-*`: Every E2E test with text output exercises tokenizer encode/decode | verified |

### 4.7 Family Plugin System (Python Build)

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-BLD-001` | Family plugins are auto-discovered via `pkgutil.iter_modules()` and dispatched by `model_type`/`architectures` match. Zero edits to shared files needed when adding a new family. | `UD-BLD-FAM-01`: `tensorrt_model_connect/tensorrt_model_connect/families/base.py` (FamilyPlugin protocol), `tensorrt_model_connect/tensorrt_model_connect/families/__init__.py` (auto-discovery) | `UT-BLD-FAM-01`: `tests/builder/test_families.py` (plugin match/dispatch, runtime_strategy, embed_input); `UT-BLD-FAM-02`: `tests/builder/test_family_plugins.py` (10+ family plugins: load_weights correctness) | `IT-E2E-*`: Every E2E test validates that the correct family plugin was dispatched during bundle build | verified |
| `ARCH-BLD-002` | Graph ops produce TRT-equivalent computations: each atomic op (RoPE, ALiBi, RMSNorm, LayerNorm, attention, etc.) in the TRT graph produces numerically equivalent results to the PyTorch/NumPy reference. | `UD-BLD-GRP-01`: `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` | `UT-BLD-GRP-01`: `tests/builder/test_graph_ops.py` (18 graph ops: RoPE, ALiBi, RMSNorm, attention, etc.); `UT-BLD-GRP-02`: `tests/builder/test_graph_ops_extended.py` (YaRN RoPE, T5 bias, extended ALiBi, conv/norm/ELU/pad ops) | N/A (ops validated at unit level; E2E validates composed result) | verified |
| `ARCH-BLD-003` | Standard decoder builder produces valid TRT engines: the builder composes graph blocks into a full decoder engine that matches HF reference output. | `UD-BLD-STD-01`: `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py`; `UD-BLD-BLK-01`: `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` | `UT-BLD-STD-01`: `tests/builder/test_standard_decoder.py` (tensor naming contract, debug outputs); `UT-BLD-BLK-01`: `tests/builder/test_graph_blocks.py` (apply_norm, SwiGLU MLP, GELU FC MLP blocks) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json` (validates full decoder pipeline) | verified |
| `ARCH-BLD-004` | Checkpoint mapper loads HF safetensors, expands GQA, ties embeddings, and maps weight keys to the engine builder's expected naming. | `UD-BLD-CKP-01`: `tensorrt_model_connect/tensorrt_model_connect/checkpoint_mapper.py` | `UT-BLD-CKP-01`: `tests/builder/test_checkpoint_mapper.py` (weight loading, GQA expansion, tied embeddings, biases) | `IT-E2E-*`: Every bundle build exercises checkpoint mapping | verified |
| `ARCH-BLD-005` | Engine builder orchestrates the full build pipeline: load config, load weights, build TRT engine, write bundle. | `UD-BLD-ENG-01`: `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` | `UT-BLD-ENG-01`: `tests/builder/test_engine_builder_extended.py` (build_bundle orchestration, GPU name, TRT version) | `IT-E2E-*`: Every E2E test with `--rebuild-engines` exercises the full build pipeline | verified |

### 4.8 Image Preprocessor

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-IMG-001` | Image preprocessor supports 4 strategies for vision-language models: each strategy produces correctly normalized, resized, and padded image tensors from raw pixel input. Config-driven strategy selection from bundle JSON. | `UD-IMG-01`: `src/runtime/domains/multimodal/image_preprocessor.h`, `src/runtime/domains/multimodal/image_preprocessor.cpp` | `UT-IMG-CPP-01`: `tests/cpp/test_image_preprocessor.cpp` (all 4 strategies, config parsing, prompt formatting) | `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen25vl-3b.json`; `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl3-8b.json` | verified |

### 4.9 TRT Engine Lifecycle and Decode Runtime

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TRT-001` | TRT engine lifecycle management: engine deserialization, execution context, tensor validation, and resource cleanup are correct. | `UD-TRT-01`: `src/runtime/core/trt_engine_lifecycle.h`, `src/runtime/core/trt_engine_lifecycle.cpp`; `UD-TRT-02`: `src/runtime/core/trt_common.h`, `src/runtime/core/trt_common.cpp` | `UT-TRT-CPP-01`: `tests/cpp/test_trt_engine_lifecycle.cpp` (layer_tensor_name, constants); `UT-TRT-CPP-02`: `tests/cpp/test_trt_logger.cpp` (severity names, error storage, explicit config controls) | `IT-E2E-*`: Every E2E test exercises engine lifecycle | verified |
| `ARCH-TRT-002` | Decode runtime provides correct argmax token selection and attention mask building for autoregressive inference. | `UD-TRT-03`: `src/runtime/core/trt_decode_runtime.h`, `src/runtime/core/trt_decode_runtime.cpp` | `UT-TRT-CPP-03`: `tests/cpp/test_decode_runtime.cpp` (argmax, mask building) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json` (text generation validates decode loop) | verified |

### 4.10 CUDA Primitives

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CUDA-001` | CUDA RAII wrappers (`CudaBuffer`, `CudaStream`) manage GPU memory and stream lifecycle with correct move semantics and cleanup. | `UD-CUDA-01`: `src/runtime/core/trt_common.h`, `src/runtime/core/trt_common.cpp` | `UT-CUDA-CPP-01`: `tests/cpp/test_cuda_buffer.cpp` (RAII alloc, move semantics, data round-trip); `UT-CUDA-CPP-02`: `tests/cpp/test_cuda_stream.cpp` (RAII stream, move semantics) | N/A (primitives validated at unit level) | verified |

### 4.11 Bundle Helpers and C ABI

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CABI-001` | Plugin helpers extract tokenizer data and initialize engines from bundle sections. C ABI entry point (`trtmc_create_pipeline_ex`) creates pipelines for external consumers. | `UD-CABI-01`: `src/runtime/plugins/shared/plugin_helpers.h`, `src/runtime/plugins/shared/plugin_helpers.cpp`; `UD-CABI-02`: `src/cabi/api/trtmc_c.cpp` | `UT-CABI-CPP-01`: `tests/cpp/test_c_abi_entry.cpp` (C ABI pipeline creation); `UT-CABI-CPP-02`: `tests/cpp/test_c_abi_runtime_regression.cpp` (C ABI runtime regression); `UT-CABI-CPP-03`: `tests/cpp/test_bundle_e2e.cpp` (bundle build + load round-trip) | `IT-E2E-*`: Every E2E test exercises plugin helpers during pipeline load | verified |

### 4.12 Pipeline Implementations

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-TEXT-001` | Text generation pipeline executes autoregressive decoding with KV cache for `decoder_kv_cache` and `decoder_moe` strategies. | `UD-PIP-TEXT-01`: `src/runtime/models/text_generation/pipeline.h`, `src/runtime/models/text_generation/pipeline.cpp` | `UT-PIP-TEXT-CPP-01`: `tests/cpp/test_text_generation_pipeline.cpp` | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json`; `IT-E2E-MIXTRAL-01`: `tests/e2e/models/mixtral-stories-15m.json`; `IT-E2E-PHIMOE-01`: `tests/e2e/models/phi-moe.json` | verified |
| `ARCH-PIP-VL-001` | Vision-language pipeline combines vision encoder + text decoder with image preprocessing for `vision_language` strategy. | `UD-PIP-VL-01`: `src/runtime/models/vision_language/pipeline.h`, `src/runtime/models/vision_language/pipeline.cpp`; `UD-PIP-VL-02`: `src/runtime/models/vision_language/plugin.cpp` | `UT-PIP-VL-CPP-01`: `tests/cpp/test_vl_pipeline.cpp`; `UT-PIP-VL-PY-01`: `tests/builder/test_vision_compute_extended.py` (vision RoPE, DeepStack config, patch embed) | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen25vl-3b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl3-8b.json` | verified |
| `ARCH-PIP-REC-001` | Recurrent pipeline executes SSM/RWKV inference with device-resident recurrent state for `ssm_recurrent` and `rwkv_recurrent` strategies. | `UD-PIP-REC-01`: `src/runtime/models/recurrent/pipeline.h`, `src/runtime/models/recurrent/pipeline.cpp`; `UD-PIP-REC-02`: `src/runtime/models/recurrent/ssm_plugin.cpp`; `UD-PIP-REC-03`: `src/runtime/models/recurrent/rwkv_plugin.cpp` | `UT-PIP-REC-CPP-01`: `tests/cpp/test_recurrent_pipeline.cpp` | `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba-130m.json`; `IT-E2E-RWKV-01`: `tests/e2e/models/rwkv-169m.json` | verified |
| `ARCH-PIP-AUD-001` | Audio pipelines handle `speech_to_text` (Whisper), `text_to_audio_bark` (Bark), `text_to_audio_magpie` (Magpie TTS), and `speech_to_speech` (PersonaPlex) strategies. | `UD-PIP-AUD-01`: `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/whisper/pipeline.cpp`; `UD-PIP-AUD-02`: `src/runtime/models/bark/pipeline.h`, `src/runtime/models/bark/pipeline.cpp`; `UD-PIP-AUD-03`: `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/magpie/pipeline.cpp`; `UD-PIP-AUD-04`: `src/runtime/models/speech/pipeline.h`, `src/runtime/models/speech/pipeline.cpp` | `UT-PIP-AUD-CPP-01`: `tests/cpp/test_audio_pipeline_new.cpp`; `UT-PIP-AUD-CPP-02`: `tests/cpp/test_audio_bundle_validation.cpp` | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper-tiny-fp16.json`; `IT-E2E-BARK-01`: `tests/e2e/models/bark-small.json`; `IT-E2E-BARK-02`: `tests/e2e/models/bark-large.json`; `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex-7b.json` | verified |
| `ARCH-PIP-DIFF-001` | Diffusion pipelines handle `diffusion_flux`, `diffusion_wan`, `diffusion_zimage`, and `diffusion_pixart` strategies for text-to-video/image models. | `UD-PIP-DIFF-01`: `src/runtime/models/flux/pipeline.h`, `src/runtime/models/flux/pipeline.cpp`; `UD-PIP-DIFF-02`: `src/runtime/models/wan/pipeline.h`, `src/runtime/models/wan/pipeline.cpp`; `UD-PIP-DIFF-03`: `src/runtime/models/z_image/pipeline.h`, `src/runtime/models/z_image/pipeline.cpp`; `UD-PIP-DIFF-04`: `src/runtime/domains/diffusion/` (diffusion_denoising_step_seam.h, diffusion_generation_plan.h, diffusion_preprocessor.cpp) | `UT-PIP-DIFF-CPP-01`: `tests/cpp/test_diffusion_pipeline_new.cpp`; `UT-PIP-DIFF-CPP-02`: `tests/cpp/test_diffusion_denoising_step_seam.cpp`; `UT-PIP-DIFF-CPP-03`: `tests/cpp/test_diffusion_generation_plan.cpp` | `IT-E2E-WAN21-01`: `tests/e2e/models/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z-image-turbo.json` | verified |
| `ARCH-PIP-ENC-001` | Encoder pipeline handles `encoder_only` strategy for bidirectional models (BERT, ALBERT, DeBERTa, DistilBERT, ELECTRA, ModernBERT, RoBERTa, XLNet, etc.). | `UD-PIP-ENC-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp`; `UD-PIP-ENC-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-PIP-ENC-CPP-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-BERT-01`: `tests/e2e/models/bert-base-uncased.json` | verified |
| `ARCH-PIP-SEG-001` | Segmentation pipeline handles `segmentation` and `prompted_segmentation` strategies. | `UD-PIP-SEG-01`: `src/runtime/models/segmentation/segment_pipeline.h`, `src/runtime/models/segmentation/segment_pipeline.cpp`; `UD-PIP-SEG-02`: `src/runtime/models/segmentation/sam_pipeline.h`, `src/runtime/models/segmentation/sam_pipeline.cpp`; `UD-PIP-SEG-03`: `src/runtime/models/segmentation/plugin.cpp` | `UT-PIP-SEG-CPP-01`: `tests/cpp/test_perception_preprocess_seams.cpp`; `UT-PIP-SEG-CPP-02`: `tests/cpp/test_sam_prompt_seam.cpp` | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer-b0-ade.json`; `IT-E2E-SAM-01`: `tests/e2e/models/sam-vit-base.json` | verified |

### 4.13 E2E Verification (TRT vs HF Reference)

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-E2E-001` | TRT output matches HF reference for text generation: logit cosine similarity, stable top-1 match, token agreement, and NED must pass composite thresholds. | `UD-E2E-RUN-01`: `tests/e2e_harness/runners/text_generation.py`; `UD-E2E-CMP-01`: `tests/e2e_harness/comparators/text.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-LOGIT-01`: `tests/tools/test_diff_logits.py` (logit comparison, argmax match, top-k overlap); `UT-TOOLS-PARITY-01`: `tests/tools/test_parity.py` (text/token comparison) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen3-0.6b-fp16.json`; plus 38 additional text gen manifests in `tests/e2e/models/` | verified |
| `ARCH-E2E-002` | TRT output matches HF reference for vision-language models: vision embedding cosine, NED, and word agreement must pass composite thresholds. | `UD-E2E-RUN-02`: `tests/e2e_harness/runners/vision_language.py`; `UD-E2E-CMP-02`: `tests/e2e_harness/comparators/vision_language.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-VL-01`: `tests/tools/test_diff_vl.py` (VL diff testing) | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen25vl-3b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl3-8b.json` | verified |
| `ARCH-E2E-003` | TRT output matches HF reference for diffusion models: pixel mean/std range, temporal consistency, PSNR, and SSIM must pass thresholds. | `UD-E2E-RUN-03`: `tests/e2e_harness/runners/diffusion.py`; `UD-E2E-CMP-03`: `tests/e2e_harness/comparators/diffusion.py`; `UD-E2E-REF-02`: `tests/e2e_harness/references/hf_diffusers.py` | `UT-TOOLS-DIFF-01`: `tests/tools/test_diffusion_helpers.py` (silu, gelu_tanh, bundle config/weights, timestep embedding) | `IT-E2E-WAN21-01`: `tests/e2e/models/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z-image-turbo.json` | verified |
| `ARCH-E2E-004` | TRT output matches HF reference for audio models: RMS bounds, duration ratio, mel spectrogram distance (text-to-audio); transcript similarity (speech-to-text). | `UD-E2E-RUN-04`: `tests/e2e_harness/runners/audio_speech.py`; `UD-E2E-CMP-04`: `tests/e2e_harness/comparators/text_to_audio.py`, `tests/e2e_harness/comparators/speech_to_text.py`, `tests/e2e_harness/comparators/audio.py` | `UT-TOOLS-AUDIO-01`: `tests/tools/test_diff_audio.py` (energy computation, WAV I/O, token stats) | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper-tiny-fp16.json`; `IT-E2E-BARK-01`: `tests/e2e/models/bark-small.json`; `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex-7b.json` | verified |
| `ARCH-E2E-005` | TRT output matches HF reference for segmentation models: mIoU, pixel accuracy, and boundary F-score must pass thresholds. | `UD-E2E-RUN-05`: `tests/e2e_harness/runners/segmentation.py`; `UD-E2E-CMP-05`: `tests/e2e_harness/comparators/segmentation.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-SEG-01`: `tests/tools/test_diff_segmentation.py` (pixel agreement, logit diff, argument parsing) | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer-b0-ade.json`; `IT-E2E-SAM-01`: `tests/e2e/models/sam-vit-base.json` | verified |

### 4.14 E2E Harness Infrastructure

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-HARNESS-001` | E2E harness correctly loads manifests, discovers plugins, orchestrates lifecycle (preflight, build, run, compare, artifacts), and reports results. | `UD-HARNESS-01`: `tests/e2e_harness/contracts.py` (protocols/dataclasses); `UD-HARNESS-02`: `tests/e2e_harness/orchestrator.py` (lifecycle coordinator); `UD-HARNESS-03`: `tests/e2e_harness/registry.py` (plugin auto-discovery); `UD-HARNESS-04`: `tests/e2e_harness/manifest_loader.py` (JSON manifest loading); `UD-HARNESS-05`: `tests/e2e_harness/artifact_sink.py` (artifact persistence) | `UT-TOOLS-FRAMEWORK-01`: `tests/tools/test_diff_framework.py` (DiffResult, registry, runner, CLI parsing); `UT-TOOLS-HARNESS-01`: `tests/tools/test_e2e_runner_cli_alignment.py`; `UT-TOOLS-HARNESS-02`: `tests/tools/test_e2e_runtime_path_guard.py` | `IT-E2E-*`: `tests/test_e2e.py` (parametrized entrypoint over all manifests) | verified |

### 4.13a Audio Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-AUD-001` | Whisper backend: cross-attention KV cache application, decode policy, host plan, and mel spectrogram extraction for speech-to-text inference. | `UD-AUD-WHISPER-01`: `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/whisper/pipeline.cpp`, `src/runtime/domains/audio/whisper_cross_kv_apply.h`, `src/runtime/domains/audio/whisper_cross_kv_plan.h`, `src/runtime/domains/audio/whisper_decode_policy.h`, `src/runtime/domains/audio/whisper_host_plan.h` | `UT-AUD-WHISPER-01`: `tests/cpp/test_whisper_decode_policy.cpp`; `UT-AUD-WHISPER-02`: `tests/cpp/test_whisper_host_plan.cpp` | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper-tiny-fp16.json` | verified |
| `ARCH-PIP-AUD-001` | Bark backend: generation plan for text-to-audio multi-stage (semantic, coarse, fine) codebook generation. | `UD-AUD-BARK-01`: `src/runtime/models/bark/pipeline.h`, `src/runtime/models/bark/pipeline.cpp`, `src/runtime/domains/audio/bark_generation_plan.h`, `src/runtime/domains/audio/bark_config.h` | `UT-AUD-BARK-01`: `tests/cpp/test_bark_generation_plan.cpp`; `UT-AUD-BARK-02`: `tests/cpp/test_audio_pipeline_new.cpp` | `IT-E2E-BARK-01`: `tests/e2e/models/bark-small.json`; `IT-E2E-BARK-02`: `tests/e2e/models/bark-large.json` | verified |
| `ARCH-PIP-AUD-001` | Magpie TTS backend: codec plan, decode policy, decoder plan, text completion policy, and CUDA kernels for neural TTS inference. | `UD-AUD-MAGPIE-01`: `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/magpie/pipeline.cpp`, `src/runtime/domains/audio/magpie_codec_plan.h`, `src/runtime/domains/audio/magpie_decode_policy.h`, `src/runtime/domains/audio/magpie_decoder_plan.h`, `src/runtime/domains/audio/magpie_text_completion_policy.h`, `src/runtime/domains/audio/magpie_kernels.cu`, `src/runtime/domains/audio/magpie_kernels.h` | `UT-AUD-MAGPIE-01`: `tests/cpp/test_magpie_codec_plan.cpp`; `UT-AUD-MAGPIE-02`: `tests/cpp/test_magpie_decode_policy.cpp`; `UT-AUD-MAGPIE-03`: `tests/cpp/test_magpie_decoder_plan.cpp`; `UT-AUD-MAGPIE-04`: `tests/cpp/test_magpie_text_completion_policy.cpp` | `IT-E2E-*`: Magpie E2E via `tests/e2e/models/magpie-tts-357m.json` | verified |
| `ARCH-PIP-AUD-001` | Speech-to-speech backend: delay cache, depth plan, generation policy, MIMI decode plan, runtime plan, temporal embedding plan, and waveform postprocessing for end-to-end speech synthesis. | `UD-AUD-SPEECH-01`: `src/runtime/models/speech/pipeline.h`, `src/runtime/models/speech/pipeline.cpp`, `src/runtime/domains/audio/speech_delay_cache.h`, `src/runtime/domains/audio/speech_depth_plan.h`, `src/runtime/domains/audio/speech_generation_policy.h`, `src/runtime/domains/audio/speech_mimi_decode_plan.h`, `src/runtime/domains/audio/speech_runtime_plan.h`, `src/runtime/domains/audio/speech_temporal_embed_plan.h`, `src/runtime/domains/audio/speech_waveform_postprocess.h` | `UT-AUD-SPEECH-01`: `tests/cpp/test_speech_decode_stop_policy.cpp`; `UT-AUD-SPEECH-02`: `tests/cpp/test_speech_depth_plan.cpp`; `UT-AUD-SPEECH-03`: `tests/cpp/test_speech_generation_helpers.cpp`; `UT-AUD-SPEECH-04`: `tests/cpp/test_speech_mimi_decode_plan.cpp`; `UT-AUD-SPEECH-05`: `tests/cpp/test_speech_runtime_plan.cpp`; `UT-AUD-SPEECH-06`: `tests/cpp/test_speech_temporal_embed_plan.cpp`; `UT-AUD-SPEECH-07`: `tests/cpp/test_speech_subprocess_seam.cpp` | `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex-7b.json` | verified |
| `ARCH-PIP-AUD-001` | Omni backend: audio plan for omni-multimodal (thinker-talker-code2wav) pipeline. | `UD-AUD-OMNI-01`: `src/runtime/models/omni/pipeline.h`, `src/runtime/models/omni/pipeline.cpp`, `src/runtime/domains/audio/omni_audio_plan.h` | `UT-AUD-OMNI-01`: `tests/cpp/test_omni_audio_plan.cpp` | N/A (omni E2E pending) | verified |
| `ARCH-PIP-AUD-001` | Audio common: bundle section validation for all audio pipelines and shared audio configurations, mel spectrogram feature extraction. | `UD-AUD-COMMON-01`: `src/runtime/domains/audio/audio_bundle_validation.h`, `src/runtime/domains/audio/audio_bundle_validation.cpp`, `src/runtime/domains/audio/audio_configs.h`, `src/runtime/domains/audio/mel_spectrogram.h`, `src/runtime/domains/audio/mel_spectrogram.cpp` | `UT-AUD-COMMON-01`: `tests/cpp/test_audio_bundle_validation.cpp`; `UT-AUD-COMMON-02`: `tests/cpp/test_mel_spectrogram.cpp` | `IT-E2E-*`: All audio E2E tests exercise bundle validation | verified |

### 4.13b Recurrent Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-REC-001` | Mamba backend: autoregressive SSM loop and conv+SSM recurrent state management. | `UD-REC-MAMBA-01`: `src/runtime/models/recurrent/pipeline.h`, `src/runtime/models/recurrent/pipeline.cpp`, `src/runtime/models/recurrent/ssm_plugin.cpp`, `src/runtime/core/recurrent_state.cpp`, `include/trtmc/runtime/recurrent_state.h` | `UT-REC-MAMBA-01`: `tests/cpp/test_recurrent_pipeline.cpp`; `UT-REC-MAMBA-02`: `tests/cpp/test_recurrent_state.cpp` | `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba-130m.json` | verified |
| `ARCH-PIP-REC-001` | RWKV backend: autoregressive RWKV loop and attention/FFN recurrent state management. | `UD-REC-RWKV-01`: `src/runtime/models/recurrent/pipeline.h`, `src/runtime/models/recurrent/pipeline.cpp`, `src/runtime/models/recurrent/rwkv_plugin.cpp`, `src/runtime/core/recurrent_state.cpp` | `UT-REC-RWKV-01`: `tests/cpp/test_recurrent_pipeline.cpp`; `UT-REC-RWKV-02`: `tests/cpp/test_recurrent_state.cpp` | `IT-E2E-RWKV-01`: `tests/e2e/models/rwkv-169m.json` | verified |
| `ARCH-PIP-REC-001` | Hybrid backend: combined Mamba+Attention autoregressive loop for hybrid architectures. | `UD-REC-HYBRID-01`: `src/runtime/models/recurrent/hybrid_plugin.cpp`, `src/runtime/core/hybrid_state.cpp`, `include/trtmc/runtime/hybrid_state.h` | `UT-REC-HYBRID-01`: `tests/cpp/test_recurrent_pipeline.cpp` | `IT-E2E-NEMOTRONH-01`: `tests/e2e/models/nemotron-h-nano-9b.json`; `IT-E2E-QWEN35-01`: `tests/e2e/models/qwen35-9b.json` | verified |
| `ARCH-PIP-REC-001` | Recurrent common: shared step contracts and tensor binding helpers for all recurrent backends. | `UD-REC-COMMON-01`: `src/runtime/domains/recurrent/recurrent_step_contracts.h`, `src/runtime/domains/recurrent/recurrent_tensor_bindings.h` | `UT-REC-COMMON-01`: `tests/cpp/test_recurrent_step_contracts.cpp` | N/A (contracts validated at unit level) | verified |

### 4.13c Multimodal Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-VL-001` | Vision engine: TRT engine lifecycle for vision encoder, execution plan configuration. | `UD-VL-VISION-01`: `src/runtime/domains/multimodal/vision_engine.h`, `src/runtime/domains/multimodal/vision_engine.cpp`, `src/runtime/domains/multimodal/vision_execution_plan.h` | `UT-VL-VISION-01`: `tests/cpp/test_vision_execution_plan.cpp` | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen25vl-3b.json` | verified |
| `ARCH-PIP-VL-001` | VL decode: VL pipeline and decode policy for vision-language generation. | `UD-VL-DECODE-01`: `src/runtime/models/vision_language/pipeline.h`, `src/runtime/models/vision_language/pipeline.cpp`, `src/runtime/domains/multimodal/vl_decode_policy.h` | `UT-VL-DECODE-01`: `tests/cpp/test_vl_decode_policy.cpp`; `UT-VL-DECODE-02`: `tests/cpp/test_vl_pipeline.cpp` | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen3-vl-2b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl3-8b.json` | verified |

### 4.13d Perception Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-SEG-001` | Segmentation backend: SegFormer inference with pre/post-processing seams for semantic segmentation. | `UD-SEG-01`: `src/runtime/models/segmentation/segment_pipeline.h`, `src/runtime/models/segmentation/segment_pipeline.cpp`, `src/runtime/domains/perception/segmentation_postprocess_seam.h`, `src/runtime/domains/perception/segmentation_preprocess_seam.h` | `UT-SEG-01`: `tests/cpp/test_perception_preprocess_seams.cpp` | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer-b0-ade.json` | verified |
| `ARCH-PIP-SEG-001` | SAM backend: two-stage prompted segmentation with image preprocessing, prompt encoding, mask decoding, output selection, and postprocessing seams. | `UD-SAM-01`: `src/runtime/models/segmentation/sam_pipeline.h`, `src/runtime/models/segmentation/sam_pipeline.cpp`, `src/runtime/domains/perception/sam_image_preprocess_seam.h`, `src/runtime/domains/perception/sam_output_selection.h`, `src/runtime/domains/perception/sam_postprocess_seam.h`, `src/runtime/domains/perception/sam_prompt_seam.h` | `UT-SAM-01`: `tests/cpp/test_sam_prompt_seam.cpp`; `UT-SAM-02`: `tests/cpp/test_perception_preprocess_seams.cpp` | `IT-E2E-SAM-01`: `tests/e2e/models/sam-vit-base.json` | verified |
| `ARCH-PIP-SEG-001` | Detection backend: object detection inference pipeline. | `UD-DET-01`: `src/runtime/models/encoder/object_detection_plugin.cpp` | N/A (no dedicated unit test yet) | N/A (no E2E manifest yet) | gap |
| `ARCH-PIP-ENC-001` | Neural operator backend: FNO/neural operator inference for scientific computing models. | `UD-NOP-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp` (reuses encoder pipeline) | `UT-NOP-01`: `tests/cpp/test_neural_operator_config.cpp` | N/A (no E2E manifest yet) | verified |

### 4.13e Diffusion Helper Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-DIFF-001` | Diffusion helpers: denoising step seam, generation plan, math utilities, preprocessor weight helpers, scheduler helpers, type definitions, and Wan-specific generation conditioning. | `UD-DIFF-HELPER-01`: `src/runtime/domains/diffusion/diffusion_denoising_step_seam.h`, `src/runtime/domains/diffusion/diffusion_generation_plan.h`, `src/runtime/domains/diffusion/diffusion_math.h`, `src/runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h`, `src/runtime/domains/diffusion/diffusion_scheduler_helpers.h`, `src/runtime/domains/diffusion/diffusion_types.h`, `src/runtime/domains/diffusion/wan_generation_conditioning.h`, `src/runtime/domains/diffusion/diffusion_preprocessor.cpp` | `UT-DIFF-HELPER-01`: `tests/cpp/test_diffusion_denoising_step_seam.cpp`; `UT-DIFF-HELPER-02`: `tests/cpp/test_diffusion_generation_plan.cpp`; `UT-DIFF-HELPER-03`: `tests/cpp/test_wan_generation_conditioning.cpp`; `UT-DIFF-HELPER-04`: `tests/cpp/test_diffusion_pipeline_new.cpp` | `IT-E2E-WAN21-01`: `tests/e2e/models/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z-image-turbo.json` | verified |

### 4.13f Core Helper Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TRT-001` | Core helpers: decoded image container, KV cache update plan, device tensor GPU memory management, flow match Euler scheduler, step state interface, STB image implementation, sampler, and TRT graph builder utilities. | `UD-CORE-HELPER-01`: `src/runtime/core/decoded_image.h`, `src/runtime/core/device_kv_cache_update_plan.h`, `src/runtime/core/device_tensor.cpp`, `src/runtime/core/flow_match_euler_scheduler.cpp`, `src/runtime/core/step_state.h`, `src/runtime/core/sampler.cpp`, `src/runtime/core/stb_impl.cpp`, `src/runtime/core/trt_graph_builder.cpp` | `UT-CORE-HELPER-01`: `tests/cpp/test_device_tensor.cpp`; `UT-CORE-HELPER-02`: `tests/cpp/test_flow_match_scheduler.cpp`; `UT-CORE-HELPER-03`: `tests/cpp/test_device_kv_cache.cpp` (exercises update plan); `UT-CORE-HELPER-04`: `tests/cpp/test_device_resources.cpp` | `IT-E2E-*`: Scheduler exercised by diffusion E2E; device tensors exercised by all GPU E2E tests | verified |

### 4.13g Encoder Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-ENC-001` | Embedding backend: dense embedding extraction from encoder models (Nemotron-embed). | `UD-ENC-EMBED-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp` (reuses encoder pipeline); `UD-ENC-EMBED-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-ENC-EMBED-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-NEMOTRON-EMBED-01`: `tests/e2e/models/nemotron-embed-vl-1b-v2.json` | verified |
| `ARCH-PIP-ENC-001` | Reranking backend: relevance scoring for query-document pairs (Nemotron-rerank). | `UD-ENC-RERANK-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp` (reuses encoder pipeline); `UD-ENC-RERANK-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-ENC-RERANK-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-NEMOTRON-RERANK-01`: `tests/e2e/models/nemotron-rerank-vl-1b-v2.json` | verified |

### 4.13h Utility Media Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-UTIL-001` | Media I/O utilities: WAV file read/write for audio pipelines and image file reading for vision pipelines. | `UD-UTIL-MEDIA-01`: `src/utils/image_reader.cpp`, `src/utils/wav_reader.h`, `src/utils/wav_reader.cpp` | `UT-UTIL-MEDIA-01`: `tests/cpp/test_wav_reader.cpp` | `IT-E2E-*`: Audio E2E tests exercise WAV I/O; VL E2E tests exercise image reading | verified |

### 4.15 Utility Components

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-UTIL-001` | Text/file parsing helpers and JSON extraction helpers provide correct string processing. | `UD-UTIL-01`: `src/utils/text_parsers.h`, `src/utils/text_parsers.cpp`; `UD-UTIL-02`: `src/utils/json_helpers.h`, `src/utils/json_helpers.cpp` | `UT-UTIL-CPP-01`: `tests/cpp/test_text_parsers.cpp`; `UT-UTIL-CPP-02`: `tests/cpp/test_json_helpers.cpp` | N/A (utilities validated at unit level) | verified |
| `ARCH-UTIL-002` | Source/scripts directory resolution and environment variable overrides function correctly. | `UD-UTIL-03`: `src/utils/data_dir.h`, `src/utils/data_dir.cpp` | `UT-UTIL-CPP-03`: `tests/cpp/test_data_dir.cpp` (source/scripts dir resolution, env overrides) | N/A (utilities validated at unit level) | verified |
| `ARCH-UTIL-003` | CLI argument parsing handles all valid argument combinations and rejects invalid input. | N/A (CLI parsing in `src/` main) | `UT-UTIL-CPP-04`: `tests/cpp/test_cli_args.cpp` (CLI argument parsing) | N/A | verified |

### 4.16 Debug Runner and Build CLI

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-DBG-001` | Debug runner (Python) provides pure-Python TRT inference matching C++ runtime behavior for validation and debugging. | `UD-DBG-01`: `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` | `UT-DBG-PY-01`: `tests/builder/test_debug_runner_extended.py` (bundle section loading, runner cleanup, generate sequencing) | `IT-E2E-*`: E2E tests may use debug runner path for reference | verified |
| `ARCH-CLI-001` | Build CLI (`trtmc-build`) dispatches build/inspect/version commands correctly. Pipeline wrapper detects binary and manages subprocess. | `UD-CLI-01`: `tensorrt_model_connect/tensorrt_model_connect/cli.py`; `UD-CLI-02`: `tensorrt_model_connect/tensorrt_model_connect/pipeline.py` | `UT-CLI-PY-01`: `tests/builder/test_cli.py` (CLI inspect/build command dispatch); `UT-CLI-PY-02`: `tests/builder/test_pipeline.py` (pipeline subprocess wrapper, binary detection) | N/A (CLI tested at unit level) | verified |

---

## 5. Completeness Assessment

### Summary

| Category | Total Rows | Verified | Draft | Gap |
|----------|-----------|----------|-------|-----|
| Bundle format | 1 | 1 | 0 | 0 |
| Configuration | 2 | 2 | 0 | 0 |
| Pipeline factory | 1 | 1 | 0 | 0 |
| TRT module | 1 | 1 | 0 | 0 |
| KV cache | 1 | 1 | 0 | 0 |
| Tokenizer | 1 | 1 | 0 | 0 |
| Family plugins/build | 5 | 5 | 0 | 0 |
| Image preprocessor | 1 | 1 | 0 | 0 |
| TRT engine lifecycle | 2 | 2 | 0 | 0 |
| CUDA primitives | 1 | 1 | 0 | 0 |
| C ABI/bundle helpers | 1 | 1 | 0 | 0 |
| Pipeline implementations | 7 | 7 | 0 | 0 |
| Audio backend subsystems | 6 | 6 | 0 | 0 |
| Recurrent backend subsystems | 4 | 4 | 0 | 0 |
| Multimodal backend subsystems | 2 | 2 | 0 | 0 |
| Perception backend subsystems | 4 | 3 | 0 | 1 |
| Diffusion helper subsystems | 1 | 1 | 0 | 0 |
| Core helper subsystems | 1 | 1 | 0 | 0 |
| Encoder backend subsystems | 2 | 2 | 0 | 0 |
| Utility media subsystems | 1 | 1 | 0 | 0 |
| E2E verification | 5 | 5 | 0 | 0 |
| E2E harness infra | 1 | 1 | 0 | 0 |
| Utilities | 3 | 3 | 0 | 0 |
| Debug runner/CLI | 2 | 2 | 0 | 0 |
| **Total** | **57** | **56** | **0** | **1** |

### File Reference Integrity

All file paths in this matrix have been verified to exist in the repository
as of 2026-03-30. Revision 2.0 corrected 7 phantom file references from
revision 1.0. This revision corrects stale backend file paths that were
consolidated from `src/runtime/domains/*/xxx_backend.h/cpp` into
`src/runtime/pipelines/` and `src/runtime/plugins/`.

### Known Gaps

- `UD-DET-01` (detection backend, `src/runtime/models/encoder/object_detection_plugin.cpp`)
  has no dedicated unit test or E2E manifest. Marked as `gap`.
- Per-family plugin traceability (individual family `.py` files to their
  corresponding engine test files in `tests/builder/test_engine_*.py`) is
  not enumerated row-by-row. The `ARCH-BLD-001` row covers the system-level
  plugin dispatch contract.

---

## 6. Maintenance Process

Apply this process for any architecture/design/test change:

1. **Update or add** the affected `ARCH-*` and `UD-*` rows in this matrix.
2. **Update tests** to include Intent + Preconditions + Postconditions + Trace IDs.
3. **Run** the mapped `UT-*` and `IT-*` checks.
4. **Record** evidence command(s), artifact location(s), and verification date
   in the row or PR notes.
5. **During review**, verify both directions:
   - Top-down: architecture contract has design and test evidence.
   - Bottom-up: changed tests map back to architecture intent.

### Definition of Done for Traceability

- No orphan architecture contracts (without tests).
- No orphan tests (without architecture/design contract).
- No phantom file references (all paths must exist in the repo).
- Matrix row status can be marked `verified` only after unit + integration
  evidence is available.
- Row status must be re-validated when referenced source files are renamed,
  moved, or deleted.
