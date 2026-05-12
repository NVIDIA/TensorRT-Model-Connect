# Source ownership manifest for the core runtime target.
#
# Keep the top-level CMakeLists focused on target wiring. Source membership
# lives here so runtime/core, runtime/domains, runtime/plugins, and pipelines
# can evolve without expanding the root build file.

set(TRTMC_CORE_SOURCES
  # Bundle and C ABI
  src/bundle/bundle_format.cpp
  src/bundle/bundle_view.cpp
  src/cabi/api/trtmc_c.cpp

  # Core runtime infrastructure
  src/runtime/core/trt_common.cpp
  src/runtime/core/gpu_matmul.cpp
  src/runtime/core/cuda_common.cpp
  src/runtime/backend/backend_loader.cpp
  src/runtime/backend/trt_version.cpp
  src/runtime/core/trt_engine_lifecycle.cpp
  src/runtime/core/trt_decode_runtime.cpp
  src/runtime/core/sampler.cpp
  src/runtime/core/device_kv_cache.cpp
  src/runtime/core/device_tensor.cpp
  src/runtime/core/kv_cache.cpp
  src/runtime/core/triattention_kv_cache.cpp
  src/runtime/core/recurrent_state.cpp
  src/runtime/core/hybrid_state.cpp
  src/runtime/core/chat_template.cpp
  src/runtime/core/flow_match_euler_scheduler.cpp
  src/runtime/core/stb_impl.cpp

  # Domain-specific components
  src/runtime/domains/multimodal/image_preprocessor.cpp
  src/runtime/domains/multimodal/vision_engine.cpp
  src/runtime/domains/audio/audio_types.cpp
  src/runtime/domains/audio/mel_spectrogram.cpp
  src/runtime/domains/audio/audio_bundle_validation.cpp
  src/runtime/domains/diffusion/diffusion_preprocessor.cpp

  # Tokenizers
  src/tokenizer/vocab_tokenizer.cpp
  src/tokenizer/ipa_tokenizer.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/tokenizer/wordpiece_tokenizer.cpp
  src/tokenizer/unigram_tokenizer.cpp

  # Utilities
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/utils/wav_reader.cpp
  src/utils/image_reader.cpp

  # Config registry and layered config resolution
  src/runtime/config/schema_registry.cpp
  src/runtime/config/config_bundle.cpp
  src/runtime/config/cli_support.cpp

  # Pipeline registry and plugin dispatch
  src/runtime/registry/pipeline_factory.cpp
  src/runtime/registry/pipeline_registry.cpp
  src/runtime/registry/pipeline_plugin.cpp

  # Plugin shared helpers
  src/runtime/plugins/shared/plugin_helpers.cpp
  src/runtime/plugins/shared/diffusion_helpers.cpp
  src/runtime/plugins/shared/audio_helpers.cpp

  # Pipeline implementations
  src/runtime/pipelines/text_generation_pipeline.cpp
  src/runtime/pipelines/recurrent_pipeline.cpp
  src/runtime/pipelines/encoder_pipeline.cpp
  src/runtime/pipelines/patchtst_pipeline.cpp
  src/runtime/pipelines/patchtsmixer_pipeline.cpp
  src/runtime/pipelines/timesfm_pipeline.cpp
  src/runtime/pipelines/chronos_bolt_pipeline.cpp
  src/runtime/pipelines/segment_pipeline.cpp
  src/runtime/pipelines/sam_pipeline.cpp
  src/runtime/pipelines/vl_pipeline.cpp
  src/runtime/pipelines/flux_pipeline.cpp
  src/runtime/pipelines/ltx_video_pipeline.cpp
  src/runtime/pipelines/wan_pipeline.cpp
  src/runtime/pipelines/z_image_pipeline.cpp
  src/runtime/pipelines/whisper_pipeline.cpp
  src/runtime/pipelines/rnnt_pipeline.cpp
  src/runtime/pipelines/bark_pipeline.cpp
  src/runtime/pipelines/magpie_pipeline.cpp
  src/runtime/pipelines/speech_pipeline.cpp
  src/runtime/pipelines/omni_pipeline.cpp
  src/runtime/pipelines/pixart_pipeline.cpp
  src/runtime/pipelines/pixart_torchtrt_pipeline.cpp
)

set(TRTMC_CUDA_KERNEL_SOURCES
  src/runtime/domains/audio/magpie_kernels.cu
  src/runtime/core/argmax_kernel.cu
  src/runtime/core/sparse_multinomial_kernel.cu
  src/runtime/core/triattention_kernels.cu
)

set(TRTMC_TVM_FFI_MODULE_LOADER_SOURCE
  src/runtime/plugins/tvm_ffi/tvm_ffi_module_loader.cpp
)

set(TRTMC_TVM_FFI_PLUGIN_SOURCES
  src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_plugin.cpp
  src/runtime/plugins/tvm_ffi/tvm_ffi_kernel_creator.cpp
)
