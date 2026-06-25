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
- Python build package (`python/tensorrt_model_connect/`)
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
| `ARCH-BDL-001` | Bundle format read/write roundtrip: `.trtfb` bundles must survive write-then-read without data loss; section headers, magic bytes, and payload integrity are preserved. | `UD-BDL-01`: `src/bundle/bundle_format.h`, `src/bundle/bundle_format.cpp`; `UD-BDL-02`: `python/tensorrt_model_connect/bundle_writer.py` | `UT-BDL-CPP-01`: `tests/cpp/test_bundle_format.cpp` (magic, section parsing, round-trip); `UT-BDL-PY-01`: `tests/builder/test_bundle_writer.py` (bundle format round-trip, section integrity) | `IT-E2E-*`: Every E2E test in `tests/test_e2e.py` exercises bundle read/write (all 197 manifests in `tests/e2e/models/`) | verified |

### 4.2 Configuration

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CFG-001` | `BaseConfig` (`parse_base_config()`) parses runtime strategies declared by model manifests, including family-owned seq2seq keys such as `t5_text_to_text`, `bart_seq2seq_encoder_decoder`, `m2m_100_seq2seq_encoder_decoder`, and `marian_translation`, from bundle JSON config. | `UD-CFG-01`: `include/trtmc/runtime/pipeline_plugin.h`, `src/runtime/registry/pipeline_plugin.cpp` | `UT-CFG-CPP-01`: `tests/cpp/test_pipeline_registry.cpp` (strategy registration, lookup); `UT-CFG-CPP-02`: `tests/cpp/test_pipeline_api.cpp` (pipeline creation via factory) | `IT-E2E-*`: Every E2E test validates config parsing as part of bundle load | verified |
| `ARCH-CFG-002` | `ModelConfig` (Python) correctly parses HuggingFace `config.json` for all supported model families, extracting architecture parameters needed for engine building. | `UD-CFG-02`: `python/tensorrt_model_connect/config.py` | `UT-CFG-PY-01`: `tests/builder/test_config.py` (ModelConfig parsing, VL text_config merge, edge cases) | N/A (build-time only) | verified |

### 4.3 Pipeline Factory and Dispatch

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-FAC-001` | `PipelineFactory` dispatches to the correct pipeline implementation based on `runtime_strategy` in the bundle config. Each strategy routes to a distinct pipeline class. | `UD-FAC-01`: `src/runtime/registry/pipeline_factory.cpp`, `include/trtmc/runtime/pipeline_factory.h`; `UD-FAC-02`: `src/cabi/api/trtmc_c.cpp` (C ABI entry point) | `UT-FAC-CPP-01`: `tests/cpp/test_pipeline_api.cpp` (pipeline creation via factory); `UT-FAC-CPP-02`: `tests/cpp/test_c_abi_entry.cpp` (C ABI dispatch); `UT-FAC-CPP-03`: `tests/cpp/test_pipeline_registry.cpp` (registry lookup) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json` via `tests/test_e2e.py` (decoder_kv_cache); `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba/manifests/mamba-130m.json` (mamba_ssm_recurrent); `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen3-vl-2b.json` (vision_language) | verified |

### 4.4 TRT Module

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-MOD-001` | `TrtModule` wraps TRT engine execution: manages engine deserialization, execution context creation, tensor binding, and synchronous inference dispatch. All pipeline backends use `TrtModule` as their engine interface. | `UD-MOD-01`: `include/trtmc/runtime/trt_module.h`, `src/runtime/backend/trt_module_impl.cpp` | `UT-MOD-CPP-01`: `tests/cpp/test_trt_module.cpp` (module construction, tensor binding, lifecycle) | `IT-E2E-*`: Every E2E test exercises `TrtModule` through pipeline execution | verified |

### 4.5 KV Cache

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-KVC-001` | KV cache manages autoregressive state: cache helpers support append, position tracking, mask progression, and reset. Concrete decoder cache execution is model-owned. | `UD-KVC-01`: `include/trtmc/runtime/kv_cache.h`, `src/runtime/core/kv_cache.cpp` | `UT-KVC-CPP-01`: `tests/cpp/test_kv_cache_new.cpp` (cache construction and update behavior); `UT-KVC-PY-01`: `tests/builder/test_cache_state_machine.py` (position, mask, cache append/shift logic) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json` (autoregressive text gen validates cache correctness) | verified |

### 4.6 Tokenizer

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TOK-001` | Tokenizer encode/decode roundtrip: `VocabTokenizer` (vocabulary lookup), `BpeTokenizer` (BPE), `WordPieceTokenizer`, `UnigramTokenizer`, and `IpaTokenizer` (IPA phoneme) all support lossless encode-then-decode for in-vocabulary text. HuggingFace tokenizer integration via Python subprocess is handled at the pipeline level. `add_special_tokens` is controlled by bundle config field. | `UD-TOK-01`: `src/tokenizer/vocab_tokenizer.cpp`; `UD-TOK-02`: `src/tokenizer/bpe_tokenizer.cpp`; `UD-TOK-03`: `src/tokenizer/wordpiece_tokenizer.cpp`; `UD-TOK-04`: `src/tokenizer/unigram_tokenizer.cpp`; `UD-TOK-05`: `src/tokenizer/ipa_tokenizer.cpp` | `UT-TOK-CPP-01`: `tests/cpp/test_vocab_tokenizer.cpp` (encode/decode, round-trip, case insensitivity); `UT-TOK-CPP-02`: `tests/cpp/test_bpe_tokenizer.cpp` (BPE encode/decode); `UT-TOK-CPP-03`: `tests/cpp/test_bpe_golden.cpp` (BPE golden reference); `UT-TOK-CPP-04`: `tests/cpp/test_wordpiece_tokenizer.cpp` (WordPiece encode/decode); `UT-TOK-CPP-05`: `tests/cpp/test_wordpiece_golden.cpp` (WordPiece golden reference); `UT-TOK-CPP-06`: `tests/cpp/test_unigram_tokenizer.cpp` (Unigram encode/decode); `UT-TOK-CPP-07`: `tests/cpp/test_unigram_golden.cpp` (Unigram golden reference); `UT-TOK-CPP-08`: `tests/cpp/test_ipa_tokenizer.cpp` (IPA phoneme tokenizer) | `IT-E2E-*`: Every E2E test with text output exercises tokenizer encode/decode | verified |

### 4.7 Family Plugin System (Python Build)

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-BLD-001` | Family plugins are auto-discovered via `pkgutil.iter_modules()` and dispatched by `model_type`/`architectures` match. Zero edits to shared files needed when adding a new family. | `UD-BLD-FAM-01`: `python/tensorrt_model_connect/families/base.py` (FamilyPlugin protocol), `python/tensorrt_model_connect/families/__init__.py` (auto-discovery) | `UT-BLD-FAM-01`: `tests/builder/test_families.py` (plugin match/dispatch, runtime_strategy, embed_input); `UT-BLD-FAM-02`: `tests/builder/test_family_plugins.py` (10+ family plugins: load_weights correctness) | `IT-E2E-*`: Every E2E test validates that the correct family plugin was dispatched during bundle build | verified |
| `ARCH-BLD-002` | Graph ops produce TRT-equivalent computations: each atomic op (RoPE, ALiBi, RMSNorm, LayerNorm, attention, etc.) in the TRT graph produces numerically equivalent results to the PyTorch/NumPy reference. | `UD-BLD-GRP-01`: `python/tensorrt_model_connect/graph_ops.py` | `UT-BLD-GRP-01`: `tests/builder/test_graph_ops.py` (18 graph ops: RoPE, ALiBi, RMSNorm, attention, etc.); `UT-BLD-GRP-02`: `tests/builder/test_graph_ops_extended.py` (YaRN RoPE, T5 bias, extended ALiBi, conv/norm/ELU/pad ops) | N/A (ops validated at unit level; E2E validates composed result) | verified |
| `ARCH-BLD-003` | Standard decoder builders produce valid TRT engines: family-local builders compose graph blocks into full decoder engines that match HF reference output. | `UD-BLD-STD-01`: `python/tensorrt_model_connect/families/qwen/standard_decoder_builder.py`; `UD-BLD-BLK-01`: `python/tensorrt_model_connect/graph_blocks.py` | `UT-BLD-STD-01`: `tests/builder/test_standard_decoder.py` (tensor naming contract, debug outputs); `UT-BLD-BLK-01`: `tests/builder/test_graph_blocks.py` (apply_norm, SwiGLU MLP, GELU FC MLP blocks) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json` (validates full decoder pipeline) | verified |
| `ARCH-BLD-004` | Checkpoint mapper loads HF safetensors, expands GQA, ties embeddings, and maps weight keys to the engine builder's expected naming. | `UD-BLD-CKP-01`: `python/tensorrt_model_connect/checkpoint_mapper.py` | `UT-BLD-CKP-01`: `tests/builder/test_checkpoint_mapper.py` (weight loading, GQA expansion, tied embeddings, biases) | `IT-E2E-*`: Every bundle build exercises checkpoint mapping | verified |
| `ARCH-BLD-005` | Engine builder orchestrates the full build pipeline: load config, load weights, build TRT engine, write bundle. | `UD-BLD-ENG-01`: `python/tensorrt_model_connect/engine_builder.py` | `UT-BLD-ENG-01`: `tests/builder/test_engine_builder_extended.py` (build_bundle orchestration, GPU name, TRT version) | `IT-E2E-*`: Every E2E test with `--rebuild-engines` exercises the full build pipeline | verified |

### 4.8 Image Preprocessor

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-IMG-001` | Each vision-language family owns its image preprocessing strategies, normalization, resizing, padding, and prompt formatting. | `UD-IMG-01`: `src/runtime/models/qwen_vl/image_preprocessor.h`, `src/runtime/models/qwen_vl/image_preprocessor.cpp`; `src/runtime/models/internvl/image_preprocessor.h`, `src/runtime/models/internvl/image_preprocessor.cpp` | `UT-IMG-CPP-01`: `tests/cpp/models/qwen_vl/test_qwen_vl_vl_pipeline.cpp`; `tests/cpp/models/internvl/test_internvl_vl_pipeline.cpp` | `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen25vl-3b.json`; `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen3-vl-2b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl/manifests/internvl3-8b.json` | verified |

### 4.9 TRT Engine Lifecycle and Decode Runtime

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TRT-001` | TRT engine lifecycle management: engine deserialization, execution context, tensor validation, and resource cleanup are correct. | `UD-TRT-01`: `src/runtime/core/trt_engine_lifecycle.h`, `src/runtime/core/trt_engine_lifecycle.cpp`; `UD-TRT-02`: `src/runtime/core/trt_common.h`, `src/runtime/core/trt_common.cpp` | `UT-TRT-CPP-01`: `tests/cpp/test_trt_engine_lifecycle.cpp` (layer_tensor_name, constants); `UT-TRT-CPP-02`: `tests/cpp/test_trt_logger.cpp` (severity names, error storage, explicit config controls) | `IT-E2E-*`: Every E2E test exercises engine lifecycle | verified |
| `ARCH-TRT-002` | Decode runtime provides correct argmax token selection and attention mask building for autoregressive inference. | `UD-TRT-03`: `src/runtime/core/trt_decode_runtime.h`, `src/runtime/core/trt_decode_runtime.cpp` | `UT-TRT-CPP-03`: `tests/cpp/test_decode_runtime.cpp` (argmax, mask building) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json` (text generation validates decode loop) | verified |

### 4.10 CUDA Primitives

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CUDA-001` | CUDA RAII wrappers (`CudaBuffer`, `CudaStream`) manage GPU memory and stream lifecycle with correct move semantics and cleanup. | `UD-CUDA-01`: `src/runtime/core/trt_common.h`, `src/runtime/core/trt_common.cpp` | `UT-CUDA-CPP-01`: `tests/cpp/test_cuda_buffer.cpp` (RAII alloc, move semantics, data round-trip); `UT-CUDA-CPP-02`: `tests/cpp/test_cuda_stream.cpp` (RAII stream, move semantics) | N/A (primitives validated at unit level) | verified |

### 4.11 Bundle Helpers and C ABI

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-CABI-001` | Model-local plugin helpers extract tokenizer data and initialize engines from bundle sections. C ABI entry point (`trtmc_create_pipeline_ex`) creates pipelines for external consumers. | `UD-CABI-01`: `src/runtime/models/<model>/plugin_helpers.h`, `src/runtime/models/<model>/plugin_helpers.cpp`; `UD-CABI-02`: `src/cabi/api/trtmc_c.cpp` | `UT-CABI-CPP-01`: `tests/cpp/test_c_abi_entry.cpp` (C ABI pipeline creation); `UT-CABI-CPP-02`: `tests/cpp/test_c_abi_runtime_regression.cpp` (C ABI runtime regression); `UT-CABI-CPP-03`: `tests/cpp/test_bundle_e2e.cpp` (bundle build + load round-trip) | `IT-E2E-*`: Every E2E test exercises its model-local plugin helpers during pipeline load | verified |

### 4.12 Pipeline Implementations

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-TEXT-001` | Text generation pipeline executes autoregressive decoding with KV cache for `decoder_kv_cache` and `decoder_moe` strategies. | `UD-PIP-TEXT-01`: `src/runtime/models/<family>/pipeline.h`, `src/runtime/models/<family>/pipeline.cpp` | `UT-PIP-TEXT-CPP-01`: `tests/cpp/models/llama/test_llama_pipeline.cpp` | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json`; `IT-E2E-MIXTRAL-01`: `tests/e2e/models/mixtral/manifests/mixtral-stories-15m.json`; `IT-E2E-PHIMOE-01`: `tests/e2e/models/phi_moe/manifests/phi-moe.json` | verified |
| `ARCH-PIP-VL-001` | Vision-language families combine vision encoder + text decoder with image preprocessing in model-owned runtime plugins. | `UD-PIP-VL-01`: `src/runtime/models/qwen_vl/pipeline.h`, `src/runtime/models/qwen_vl/pipeline.cpp`; `src/runtime/models/internvl/pipeline.h`, `src/runtime/models/internvl/pipeline.cpp`; `UD-PIP-VL-02`: `src/runtime/models/qwen_vl/plugin.cpp`, `src/runtime/models/internvl/plugin.cpp` | `UT-PIP-VL-CPP-01`: `tests/cpp/models/qwen_vl/test_qwen_vl_vl_pipeline.cpp`; `tests/cpp/models/internvl/test_internvl_vl_pipeline.cpp`; `UT-PIP-VL-PY-01`: `tests/builder/test_vision_compute_extended.py` (vision RoPE, DeepStack config, patch embed) | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen25vl-3b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl/manifests/internvl3-8b.json` | verified |
| `ARCH-PIP-REC-001` | Recurrent pipeline executes SSM/RWKV inference with device-resident family-owned recurrent state for `mamba_ssm_recurrent` and `rwkv_recurrent` strategies. | `UD-PIP-REC-01`: `src/runtime/models/mamba/pipeline.h`, `src/runtime/models/mamba/pipeline.cpp`, `src/runtime/models/rwkv/pipeline.h`, `src/runtime/models/rwkv/pipeline.cpp`; `UD-PIP-REC-02`: `src/runtime/models/mamba/plugin.cpp`, `src/runtime/models/mamba/recurrent_state.h`, `src/runtime/models/mamba/recurrent_state.cpp`; `UD-PIP-REC-03`: `src/runtime/models/rwkv/plugin.cpp`, `src/runtime/models/rwkv/recurrent_state.h`, `src/runtime/models/rwkv/recurrent_state.cpp` | `UT-PIP-REC-CPP-01`: `tests/cpp/models/mamba/test_mamba_recurrent_pipeline.cpp`, `tests/cpp/models/rwkv/test_rwkv_recurrent_pipeline.cpp` | `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba/manifests/mamba-130m.json`; `IT-E2E-RWKV-01`: `tests/e2e/models/rwkv/manifests/rwkv-169m.json` | verified |
| `ARCH-PIP-AUD-001` | Audio pipelines handle `speech_to_text` (Whisper), `text_to_audio_bark` (Bark), `text_to_audio_magpie` (Magpie TTS), and `speech_to_speech` (PersonaPlex) strategies. | `UD-PIP-AUD-01`: `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/whisper/pipeline.cpp`; `UD-PIP-AUD-02`: `src/runtime/models/bark/pipeline.h`, `src/runtime/models/bark/pipeline.cpp`; `UD-PIP-AUD-03`: `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/magpie/pipeline.cpp`; `UD-PIP-AUD-04`: `src/runtime/models/speech/pipeline.h`, `src/runtime/models/speech/pipeline.cpp` | `UT-PIP-AUD-CPP-01`: `tests/cpp/test_audio_pipeline_new.cpp`; `UT-PIP-AUD-CPP-02`: `tests/cpp/test_audio_bundle_validation.cpp` | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper/manifests/whisper-tiny-fp16.json`; `IT-E2E-BARK-01`: `tests/e2e/models/bark/manifests/bark-small.json`; `IT-E2E-BARK-02`: `tests/e2e/models/bark/manifests/bark-large.json`; `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex/manifests/personaplex-7b.json` | verified |
| `ARCH-PIP-DIFF-001` | Diffusion pipelines handle `diffusion_flux`, `diffusion_wan`, `diffusion_zimage`, and `diffusion_pixart` strategies for text-to-video/image models. | `UD-PIP-DIFF-01`: `src/runtime/models/flux/pipeline.h`, `src/runtime/models/flux/pipeline.cpp`, `src/runtime/models/flux/diffusion_helpers.cpp`, `src/runtime/models/flux/flux_generation_plan.h`, `src/runtime/models/flux/flux_denoising_step_seam.h`; `UD-PIP-DIFF-02`: `src/runtime/models/wan/pipeline.h`, `src/runtime/models/wan/pipeline.cpp`, `src/runtime/models/wan/diffusion_helpers.cpp`, `src/runtime/models/wan/wan_generation_plan.h`, `src/runtime/models/wan/wan_denoising_step_seam.h`; `UD-PIP-DIFF-03`: `src/runtime/models/z_image/pipeline.h`, `src/runtime/models/z_image/pipeline.cpp`, `src/runtime/models/z_image/diffusion_helpers.cpp`; `UD-PIP-DIFF-04`: `src/runtime/models/pixart/pipeline.h`, `src/runtime/models/pixart/pipeline.cpp`, `src/runtime/models/pixart/diffusion_helpers.cpp`, `src/runtime/models/pixart/pixart_generation_plan.h`, `src/runtime/models/pixart/pixart_denoising_step_seam.h` | `UT-PIP-DIFF-CPP-01`: `tests/cpp/models/flux/test_flux_pipeline.cpp`; `UT-PIP-DIFF-CPP-02`: `tests/cpp/models/flux/test_flux_denoising_step_seam.cpp`, `tests/cpp/models/wan/test_wan_denoising_step_seam.cpp`, `tests/cpp/models/pixart/test_pixart_denoising_step_seam.cpp`; `UT-PIP-DIFF-CPP-03`: `tests/cpp/models/flux/test_flux_generation_plan.cpp`, `tests/cpp/models/wan/test_wan_generation_plan.cpp` | `IT-E2E-WAN21-01`: `tests/e2e/models/wan_t2v/manifests/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux/manifests/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z_image/manifests/z-image-turbo.json` | verified |
| `ARCH-PIP-ENC-001` | Encoder pipeline handles `encoder_only` strategy for bidirectional models (BERT, ALBERT, DeBERTa, DistilBERT, ELECTRA, ModernBERT, RoBERTa, XLNet, etc.). | `UD-PIP-ENC-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp`; `UD-PIP-ENC-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-PIP-ENC-CPP-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-BERT-01`: `tests/e2e/models/bert/manifests/bert-base-uncased.json` | verified |
| `ARCH-PIP-SEG-001` | Segmentation pipeline handles `segmentation` and `prompted_segmentation` strategies through model-owned runtime plugins. | `UD-PIP-SEG-01`: `src/runtime/models/segformer/segment_pipeline.h`, `src/runtime/models/segformer/segment_pipeline.cpp`; `UD-PIP-SEG-02`: `src/runtime/models/sam/sam_pipeline.h`, `src/runtime/models/sam/sam_pipeline.cpp`; `UD-PIP-SEG-03`: `src/runtime/models/sam3/sam3_pipeline.h`, `src/runtime/models/sam3/sam3_pipeline.cpp` | `UT-PIP-SEG-CPP-01`: `tests/cpp/models/segformer/test_segformer_preprocess_seam.cpp`; `UT-PIP-SEG-CPP-02`: `tests/cpp/models/sam/test_sam_prompt_seam.cpp`; `UT-PIP-SEG-CPP-03`: `tests/cpp/models/sam3/test_sam3_pipeline.cpp` | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer/manifests/segformer-b0-ade.json`; `IT-E2E-SAM-01`: `tests/e2e/models/sam/manifests/sam-vit-base.json` | verified |

### 4.13 E2E Verification (TRT vs HF Reference)

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-E2E-001` | TRT output matches HF reference for text generation: logit cosine similarity, stable top-1 match, token agreement, and NED must pass composite thresholds. | `UD-E2E-RUN-01`: `tests/e2e_harness/runners/text_generation.py`; `UD-E2E-CMP-01`: `tests/e2e_harness/comparators/text.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-LOGIT-01`: `tests/tools/test_diff_logits.py` (logit comparison, argmax match, top-k overlap); `UT-TOOLS-PARITY-01`: `tests/tools/test_parity.py` (text/token comparison) | `IT-E2E-QWEN3-01`: `tests/e2e/models/qwen/manifests/qwen3-0.6b-fp16.json`; plus 38 additional text gen manifests in `tests/e2e/models/` | verified |
| `ARCH-E2E-002` | TRT output matches HF reference for vision-language models: vision embedding cosine, NED, and word agreement must pass composite thresholds. | `UD-E2E-RUN-02`: `tests/e2e_harness/runners/vision_language.py`; `UD-E2E-CMP-02`: `tests/e2e_harness/comparators/vision_language.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-VL-01`: `tests/tools/test_diff_vl.py` (VL diff testing) | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen3-vl-2b.json`; `IT-E2E-QWEN25VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen25vl-3b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl/manifests/internvl3-8b.json` | verified |
| `ARCH-E2E-003` | TRT output matches HF reference for diffusion models: pixel mean/std range, temporal consistency, PSNR, and SSIM must pass thresholds. | `UD-E2E-RUN-03`: `tests/e2e_harness/runners/diffusion.py`; `UD-E2E-CMP-03`: `tests/e2e_harness/comparators/diffusion.py`; `UD-E2E-REF-02`: `tests/e2e_harness/references/hf_diffusers.py` | `UT-TOOLS-DIFF-01`: `tests/tools/test_diffusion_helpers.py` (silu, gelu_tanh, bundle config/weights, timestep embedding) | `IT-E2E-WAN21-01`: `tests/e2e/models/wan_t2v/manifests/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux/manifests/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z_image/manifests/z-image-turbo.json` | verified |
| `ARCH-E2E-004` | TRT output matches HF reference for audio models: RMS bounds, duration ratio, mel spectrogram distance (text-to-audio); transcript similarity (speech-to-text). | `UD-E2E-RUN-04`: `tests/e2e_harness/runners/audio_speech.py`; `UD-E2E-CMP-04`: `tests/e2e_harness/comparators/text_to_audio.py`, `tests/e2e_harness/comparators/speech_to_text.py`, `tests/e2e_harness/comparators/audio.py` | `UT-TOOLS-AUDIO-01`: `tests/tools/test_diff_audio.py` (energy computation, WAV I/O, token stats) | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper/manifests/whisper-tiny-fp16.json`; `IT-E2E-BARK-01`: `tests/e2e/models/bark/manifests/bark-small.json`; `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex/manifests/personaplex-7b.json` | verified |
| `ARCH-E2E-005` | TRT output matches HF reference for segmentation models: mIoU, pixel accuracy, and boundary F-score must pass thresholds. | `UD-E2E-RUN-05`: `tests/e2e_harness/runners/segmentation.py`; `UD-E2E-CMP-05`: `tests/e2e_harness/comparators/segmentation.py`; `UD-E2E-REF-01`: `tests/e2e_harness/references/hf_transformers.py` | `UT-TOOLS-SEG-01`: `tests/e2e/models/segformer/test_segformer_diff_segmentation.py` (pixel agreement, logit diff, argument parsing) | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer/manifests/segformer-b0-ade.json`; `IT-E2E-SAM-01`: `tests/e2e/models/sam/manifests/sam-vit-base.json` | verified |

### 4.14 E2E Harness Infrastructure

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-HARNESS-001` | E2E harness correctly loads manifests, discovers plugins, orchestrates lifecycle (preflight, build, run, compare, artifacts), and reports results. | `UD-HARNESS-01`: `tests/e2e_harness/contracts.py` (protocols/dataclasses); `UD-HARNESS-02`: `tests/e2e_harness/orchestrator.py` (lifecycle coordinator); `UD-HARNESS-03`: `tests/e2e_harness/registry.py` (plugin auto-discovery); `UD-HARNESS-04`: `tests/e2e_harness/manifest_loader.py` (JSON manifest loading); `UD-HARNESS-05`: `tests/e2e_harness/artifact_sink.py` (artifact persistence) | `UT-TOOLS-FRAMEWORK-01`: `tests/tools/test_diff_framework.py` (DiffResult, registry, runner, CLI parsing); `UT-TOOLS-HARNESS-01`: `tests/tools/test_e2e_runner_cli_alignment.py`; `UT-TOOLS-HARNESS-02`: `tests/tools/test_e2e_runtime_path_guard.py` | `IT-E2E-*`: `tests/test_e2e.py` (parametrized entrypoint over all manifests) | verified |

### 4.13a Audio Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-AUD-001` | Whisper backend: cross-attention KV cache application, decode policy, host plan, and mel spectrogram extraction for speech-to-text inference. | `UD-AUD-WHISPER-01`: `src/runtime/models/whisper/pipeline.h`, `src/runtime/models/whisper/pipeline.cpp`, `src/runtime/models/whisper/whisper_mel_spectrogram.h`, `src/runtime/models/whisper/whisper_mel_spectrogram.cpp`, `src/runtime/models/whisper/whisper_cross_kv_apply.h`, `src/runtime/models/whisper/whisper_cross_kv_plan.h`, `src/runtime/models/whisper/whisper_decode_policy.h`, `src/runtime/models/whisper/whisper_host_plan.h` | `UT-AUD-WHISPER-01`: `tests/cpp/models/whisper/test_whisper_decode_policy.cpp`; `UT-AUD-WHISPER-02`: `tests/cpp/models/whisper/test_whisper_host_plan.cpp`; `UT-AUD-WHISPER-03`: `tests/cpp/models/whisper/test_whisper_mel_spectrogram.cpp` | `IT-E2E-WHISPER-FP16-01`: `tests/e2e/models/whisper/manifests/whisper-tiny-fp16.json` | verified |
| `ARCH-PIP-AUD-001` | Bark backend: generation plan for text-to-audio multi-stage (semantic, coarse, fine) codebook generation. | `UD-AUD-BARK-01`: `src/runtime/models/bark/pipeline.h`, `src/runtime/models/bark/pipeline.cpp`, `src/runtime/domains/audio/bark_generation_plan.h`, `src/runtime/domains/audio/bark_config.h` | `UT-AUD-BARK-01`: `tests/cpp/test_bark_generation_plan.cpp`; `UT-AUD-BARK-02`: `tests/cpp/test_audio_pipeline_new.cpp` | `IT-E2E-BARK-01`: `tests/e2e/models/bark/manifests/bark-small.json`; `IT-E2E-BARK-02`: `tests/e2e/models/bark/manifests/bark-large.json` | verified |
| `ARCH-PIP-AUD-001` | Magpie TTS backend: codec plan, decode policy, decoder plan, text completion policy, and CUDA kernels for neural TTS inference. | `UD-AUD-MAGPIE-01`: `src/runtime/models/magpie/pipeline.h`, `src/runtime/models/magpie/pipeline.cpp`, `src/runtime/domains/audio/magpie_codec_plan.h`, `src/runtime/domains/audio/magpie_decode_policy.h`, `src/runtime/domains/audio/magpie_decoder_plan.h`, `src/runtime/domains/audio/magpie_text_completion_policy.h`, `src/runtime/domains/audio/magpie_kernels.cu`, `src/runtime/domains/audio/magpie_kernels.h` | `UT-AUD-MAGPIE-01`: `tests/cpp/test_magpie_codec_plan.cpp`; `UT-AUD-MAGPIE-02`: `tests/cpp/test_magpie_decode_policy.cpp`; `UT-AUD-MAGPIE-03`: `tests/cpp/test_magpie_decoder_plan.cpp`; `UT-AUD-MAGPIE-04`: `tests/cpp/test_magpie_text_completion_policy.cpp` | `IT-E2E-*`: Magpie E2E via `tests/e2e/models/magpie_tts/manifests/magpie-tts-357m.json` | verified |
| `ARCH-PIP-AUD-001` | Speech-to-speech backend: delay cache, depth plan, generation policy, MIMI decode plan, runtime plan, temporal embedding plan, and waveform postprocessing for end-to-end speech synthesis. | `UD-AUD-SPEECH-01`: `src/runtime/models/speech/pipeline.h`, `src/runtime/models/speech/pipeline.cpp`, `src/runtime/domains/audio/speech_delay_cache.h`, `src/runtime/domains/audio/speech_depth_plan.h`, `src/runtime/domains/audio/speech_generation_policy.h`, `src/runtime/domains/audio/speech_mimi_decode_plan.h`, `src/runtime/domains/audio/speech_runtime_plan.h`, `src/runtime/domains/audio/speech_temporal_embed_plan.h`, `src/runtime/domains/audio/speech_waveform_postprocess.h` | `UT-AUD-SPEECH-01`: `tests/cpp/test_speech_decode_stop_policy.cpp`; `UT-AUD-SPEECH-02`: `tests/cpp/test_speech_depth_plan.cpp`; `UT-AUD-SPEECH-03`: `tests/cpp/test_speech_generation_helpers.cpp`; `UT-AUD-SPEECH-04`: `tests/cpp/test_speech_mimi_decode_plan.cpp`; `UT-AUD-SPEECH-05`: `tests/cpp/test_speech_runtime_plan.cpp`; `UT-AUD-SPEECH-06`: `tests/cpp/test_speech_temporal_embed_plan.cpp`; `UT-AUD-SPEECH-07`: `tests/cpp/test_speech_subprocess_seam.cpp` | `IT-E2E-PERSONAPLEX-01`: `tests/e2e/models/personaplex/manifests/personaplex-7b.json` | verified |
| `ARCH-PIP-AUD-001` | Omni backend: audio plan for omni-multimodal (thinker-talker-code2wav) pipeline. | `UD-AUD-OMNI-01`: `src/runtime/models/omni/pipeline.h`, `src/runtime/models/omni/pipeline.cpp`, `src/runtime/domains/audio/omni_audio_plan.h` | `UT-AUD-OMNI-01`: `tests/cpp/test_omni_audio_plan.cpp` | N/A (omni E2E pending) | verified |
| `ARCH-PIP-AUD-001` | Audio common: shared audio-domain helper implementations are retired; generic public WAV I/O lives in `trtmc_io`, and model feature extraction is owned by each audio model. | `UD-AUD-COMMON-01`: `include/trtmc/trtmc_io.hpp`; model-owned audio helpers under `src/runtime/models/<audio-family>/` | `UT-AUD-COMMON-01`: `tests/cpp/test_trtmc_io.cpp`; model-owned audio helper tests | `IT-E2E-*`: Audio E2E tests exercise model-owned pipelines | verified |

### 4.13b Recurrent Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-REC-001` | Mamba backend: autoregressive SSM loop and conv+SSM recurrent state management. | `UD-REC-MAMBA-01`: `src/runtime/models/mamba/pipeline.h`, `src/runtime/models/mamba/pipeline.cpp`, `src/runtime/models/mamba/plugin.cpp`, `src/runtime/models/mamba/recurrent_state.h`, `src/runtime/models/mamba/recurrent_state.cpp` | `UT-REC-MAMBA-01`: `tests/cpp/models/mamba/test_mamba_recurrent_pipeline.cpp` | `IT-E2E-MAMBA-01`: `tests/e2e/models/mamba/manifests/mamba-130m.json` | verified |
| `ARCH-PIP-REC-001` | RWKV backend: autoregressive RWKV loop and attention/FFN recurrent state management. | `UD-REC-RWKV-01`: `src/runtime/models/rwkv/pipeline.h`, `src/runtime/models/rwkv/pipeline.cpp`, `src/runtime/models/rwkv/plugin.cpp`, `src/runtime/models/rwkv/recurrent_state.h`, `src/runtime/models/rwkv/recurrent_state.cpp` | `UT-REC-RWKV-01`: `tests/cpp/models/rwkv/test_rwkv_recurrent_pipeline.cpp` | `IT-E2E-RWKV-01`: `tests/e2e/models/rwkv/manifests/rwkv-169m.json` | verified |
| `ARCH-PIP-REC-001` | Hybrid backend: combined Mamba+Attention autoregressive loop for hybrid architectures. Hybrid state composition is owned by each hybrid family. | `UD-REC-HYBRID-01`: `src/runtime/models/nemotron_h/plugin.cpp`, `src/runtime/models/nemotron_h/hybrid_state.h`, `src/runtime/models/nemotron_h/hybrid_state.cpp`; `src/runtime/models/qwen3_5/plugin.cpp`, `src/runtime/models/qwen3_5/hybrid_state.h`, `src/runtime/models/qwen3_5/hybrid_state.cpp` | `UT-REC-HYBRID-01`: `tests/cpp/models/nemotron_h/test_nemotron_h_recurrent_pipeline.cpp`; `tests/cpp/models/qwen3_5/test_qwen3_5_recurrent_pipeline.cpp` | `IT-E2E-NEMOTRONH-01`: `tests/e2e/models/nemotron_h/manifests/nemotron-h-nano-9b.json`; `IT-E2E-QWEN35-01`: `tests/e2e/models/qwen3_5/manifests/qwen35-9b.json` | verified |
| `ARCH-PIP-REC-001` | Recurrent helpers: step contracts are model-owned for each recurrent family; shared recurrent domain helpers are retired. | `UD-REC-COMMON-01`: model-owned contracts: `src/runtime/models/mamba/mamba_recurrent_step_contracts.h`, `src/runtime/models/rwkv/rwkv_recurrent_step_contracts.h`, `src/runtime/models/nemotron_h/nemotron_h_recurrent_step_contracts.h`, `src/runtime/models/qwen3_5/qwen3_5_recurrent_step_contracts.h` | `UT-REC-CONTRACTS-01`: `tests/cpp/models/mamba/test_mamba_recurrent_output_initializers.cpp`, `tests/cpp/models/rwkv/test_rwkv_recurrent_output_initializers.cpp`, `tests/cpp/models/nemotron_h/test_nemotron_h_recurrent_output_initializers.cpp`, `tests/cpp/models/qwen3_5/test_qwen3_5_recurrent_output_initializers.cpp` | N/A (contracts validated at unit level) | verified |

### 4.13c Multimodal Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-VL-001` | VL decode: each vision-language family owns image-token injection, DeepStack inputs, and autoregressive stopping in its local pipeline. | `UD-VL-DECODE-01`: `src/runtime/models/qwen_vl/pipeline.h`, `src/runtime/models/qwen_vl/pipeline.cpp`; `src/runtime/models/internvl/pipeline.h`, `src/runtime/models/internvl/pipeline.cpp` | `UT-VL-DECODE-01`: `tests/cpp/models/qwen_vl/test_qwen_vl_vl_pipeline.cpp`; `UT-VL-DECODE-02`: `tests/cpp/models/internvl/test_internvl_vl_pipeline.cpp` | `IT-E2E-QWEN3VL-01`: `tests/e2e/models/qwen_vl/manifests/qwen3-vl-2b.json`; `IT-E2E-INTERNVL3-01`: `tests/e2e/models/internvl/manifests/internvl3-8b.json` | verified |

### 4.13d Perception Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-SEG-001` | Segmentation backend: SegFormer inference with model-owned pre/post-processing seams for semantic segmentation. | `UD-SEG-01`: `src/runtime/models/segformer/segment_pipeline.h`, `src/runtime/models/segformer/segment_pipeline.cpp`, `src/runtime/models/segformer/segformer_preprocess_seam.h`, `src/runtime/models/segformer/segformer_postprocess_seam.h` | `UT-SEG-01`: `tests/cpp/models/segformer/test_segformer_preprocess_seam.cpp`; `UT-SEG-02`: `tests/cpp/models/segformer/test_segformer_postprocess_seam.cpp` | `IT-E2E-SEGFORMER-01`: `tests/e2e/models/segformer/manifests/segformer-b0-ade.json` | verified |
| `ARCH-PIP-SEG-001` | SAM backend: two-stage prompted segmentation with image preprocessing, prompt encoding, mask decoding, output selection, and postprocessing seams. | `UD-SAM-01`: `src/runtime/models/sam/sam_pipeline.h`, `src/runtime/models/sam/sam_pipeline.cpp`, `src/runtime/models/sam/sam_image_preprocess_seam.h`, `src/runtime/models/sam/sam_output_selection.h`, `src/runtime/models/sam/sam_postprocess_seam.h`, `src/runtime/models/sam/sam_prompt_seam.h` | `UT-SAM-01`: `tests/cpp/models/sam/test_sam_prompt_seam.cpp`; `UT-SAM-02`: `tests/cpp/models/sam/test_sam_image_preprocess_seam.cpp` | `IT-E2E-SAM-01`: `tests/e2e/models/sam/manifests/sam-vit-base.json` | verified |
| `ARCH-PIP-SEG-001` | Detection backend: object detection inference pipeline. | `UD-DET-01`: `src/runtime/models/encoder/object_detection_plugin.cpp` | N/A (no dedicated unit test yet) | N/A (no E2E manifest yet) | gap |
| `ARCH-PIP-ENC-001` | Neural operator backend: FNO/neural operator inference for scientific computing models. | `UD-NOP-01`: model-owned encoder runtime plugin | N/A (no dedicated unit test yet) | N/A (no E2E manifest yet) | gap |

### 4.13e Diffusion Helper Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-DIFF-001` | Diffusion helpers: only math utilities stay shared; denoising step seams, generation plans, scheduler policy, batch policy, config and weight types, preprocessor section parsing, model weight-key parsing, and Wan conditioning are model-owned. | `UD-DIFF-HELPER-01`: `src/runtime/domains/diffusion/diffusion_math.h`; model-owned: `src/runtime/models/flux/flux_diffusion_types.h`, `src/runtime/models/wan/wan_diffusion_types.h`, `src/runtime/models/pixart/pixart_diffusion_types.h`, `src/runtime/models/z_image/z_image_diffusion_types.h`, `src/runtime/models/ltx_video/ltx_video_diffusion_types.h`, `src/runtime/models/qwen_image/qwen_image_diffusion_types.h`, `src/runtime/models/flux/flux_denoising_step_seam.h`, `src/runtime/models/wan/wan_denoising_step_seam.h`, `src/runtime/models/pixart/pixart_denoising_step_seam.h`, `src/runtime/models/flux/flux_batch_utils.h`, `src/runtime/models/z_image/z_image_batch_utils.h`, `src/runtime/models/qwen_image/qwen_image_batch_utils.h`, `src/runtime/models/flux/preprocessor_weights_helpers.h`, `src/runtime/models/wan/preprocessor_weights_helpers.h`, `src/runtime/models/pixart/preprocessor_weights_helpers.h`, `src/runtime/models/z_image/preprocessor_weights_helpers.h`, `src/runtime/models/ltx_video/preprocessor_weights_helpers.h`, `src/runtime/models/qwen_image/preprocessor_weights_helpers.h`, `src/runtime/models/wan/wan_generation_conditioning.h` | `UT-DIFF-HELPER-01`: `tests/cpp/models/flux/test_flux_denoising_step_seam.cpp`, `tests/cpp/models/wan/test_wan_denoising_step_seam.cpp`, `tests/cpp/models/pixart/test_pixart_denoising_step_seam.cpp`; `UT-DIFF-HELPER-02`: `tests/cpp/test_diffusion_math.cpp`; `UT-DIFF-HELPER-03`: `tests/cpp/models/wan/test_wan_generation_conditioning.cpp`; `UT-DIFF-HELPER-04`: `tests/cpp/models/flux/test_flux_batch_utils.cpp`, `tests/cpp/models/z_image/test_z_image_batch_utils.cpp`, `tests/cpp/models/qwen_image/test_qwen_image_batch_utils.cpp` | `IT-E2E-WAN21-01`: `tests/e2e/models/wan_t2v/manifests/wan21-t2v-1.3b.json`; `IT-E2E-FLUX-01`: `tests/e2e/models/flux/manifests/flux-schnell.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z_image/manifests/z-image-turbo.json` | verified |
| `ARCH-PIP-DIFF-001` | Diffusion scheduler helpers: branchy scheduler state and request fallback policy are owned by each diffusion runtime family. | `UD-DIFF-SCHED-FAMILY-01`: `src/runtime/models/flux/flux_scheduler_helpers.h`, `src/runtime/models/wan/wan_scheduler_helpers.h`, `src/runtime/models/pixart/pixart_scheduler_helpers.h`, `src/runtime/models/z_image/z_image_scheduler_helpers.h`, `src/runtime/models/ltx_video/ltx_video_scheduler_helpers.h` | `UT-DIFF-SCHED-FAMILY-01`: `tests/cpp/models/flux/test_flux_generation_plan.cpp`; `tests/cpp/models/wan/test_wan_generation_plan.cpp`; `tests/cpp/models/ltx_video/test_ltx_video_scheduler.cpp` | `IT-E2E-FLUX-01`: `tests/e2e/models/flux/manifests/flux-schnell.json`; `IT-E2E-WAN21-01`: `tests/e2e/models/wan_t2v/manifests/wan21-t2v-1.3b.json`; `IT-E2E-ZIMAGE-01`: `tests/e2e/models/z_image/manifests/z-image-turbo.json` | verified |
| `ARCH-PIP-DIFF-001` | Qwen Image scheduler: family-owned Flow Match Euler scheduler used by the Qwen Image C++ pipeline. | `UD-QWEN-IMAGE-SCHED-01`: `src/runtime/models/qwen_image/qwen_image_scheduler.h`, `src/runtime/models/qwen_image/qwen_image_scheduler.cpp` | `UT-QWEN-IMAGE-SCHED-01`: `tests/cpp/models/qwen_image/test_qwen_image_flow_match_scheduler.cpp` | `IT-E2E-QWEN-IMAGE-01`: `tests/e2e/models/qwen_image/manifests/qwen-image.json` | verified |

### 4.13f Core Helper Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-TRT-001` | Core helpers: decoded image container, device tensor GPU memory management, step state interface, STB image implementation, sampler, and TRT graph builder utilities. | `UD-CORE-HELPER-01`: `src/runtime/core/decoded_image.h`, `src/runtime/core/device_tensor.cpp`, `src/runtime/core/step_state.h`, `src/runtime/core/sampler.cpp`, `src/runtime/core/stb_impl.cpp`, `src/runtime/core/trt_graph_builder.cpp` | `UT-CORE-HELPER-01`: `tests/cpp/test_device_tensor.cpp` | `IT-E2E-*`: device tensors exercised by all GPU E2E tests | verified |

### 4.13g Encoder Backend Subsystems

| ARCH ID | Architecture Contract | UD ID + Real Files | UT Evidence (Real Test Files) | IT Evidence (Real E2E Paths) | Status |
|---------|----------------------|-------------------|-------------------------------|------------------------------|--------|
| `ARCH-PIP-ENC-001` | Embedding backend: dense embedding extraction from encoder models (Nemotron-embed). | `UD-ENC-EMBED-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp` (reuses encoder pipeline); `UD-ENC-EMBED-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-ENC-EMBED-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-NEMOTRON-EMBED-01`: `tests/e2e/models/eagle_vlm/manifests/nemotron-embed-vl-1b-v2.json` | verified |
| `ARCH-PIP-ENC-001` | Reranking backend: relevance scoring for query-document pairs (Nemotron-rerank). | `UD-ENC-RERANK-01`: `src/runtime/models/encoder/pipeline.h`, `src/runtime/models/encoder/pipeline.cpp` (reuses encoder pipeline); `UD-ENC-RERANK-02`: `src/runtime/models/encoder/plugin.cpp` | `UT-ENC-RERANK-01`: `tests/cpp/test_encoder_pipeline.cpp` | `IT-E2E-NEMOTRON-RERANK-01`: `tests/e2e/models/eagle_vlm/manifests/nemotron-rerank-vl-1b-v2.json` | verified |

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
| `ARCH-DBG-001` | Debug runner (Python) provides pure-Python TRT inference matching C++ runtime behavior for validation and debugging. | `UD-DBG-01`: `python/tensorrt_model_connect/debug_runner.py` | `UT-DBG-PY-01`: `tests/builder/test_debug_runner_extended.py` (bundle section loading, runner cleanup, generate sequencing) | `IT-E2E-*`: E2E tests may use debug runner path for reference | verified |
| `ARCH-CLI-001` | Unified CLI (`trtmc`) dispatches build/inspect/version commands correctly. Pipeline wrapper detects binary and manages subprocess. | `UD-CLI-01`: `python/tensorrt_model_connect/build_cli.py`; `UD-CLI-02`: `python/tensorrt_model_connect/pipeline.py` | `UT-CLI-PY-01`: `tests/builder/test_cli.py` (CLI inspect/build command dispatch); `UT-CLI-PY-02`: `tests/builder/test_pipeline.py` (pipeline subprocess wrapper, binary detection) | N/A (CLI tested at unit level) | verified |

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
`src/runtime/models/` and `src/runtime/plugins/`.

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
