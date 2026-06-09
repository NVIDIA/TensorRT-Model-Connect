cmake_minimum_required(VERSION 3.20)
project(tensorrt_model_connect VERSION 0.1.0 LANGUAGES CXX)

# Enable CUDA for GPU kernels (magpie_kernels.cu)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(TRTMC_ENABLE_TRT "Enable TensorRT backend DSO integration" ON)
option(TRTMC_BUILD_BACKEND_TRT "Build standard TensorRT backend DSO" ON)
option(TRTMC_BUILD_TESTS "Build C++ unit test executables" ON)
option(TRTMC_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL "Enable optional libtorch-backed multinomial sampler" ON)
set(TRTMC_TRT_BACKEND_ABI "" CACHE STRING
  "TensorRT ABI suffix for the standard backend DSO, e.g. 10_15. Empty means auto-detect from headers.")

# CUDA is required by the core runtime. TensorRT is optional at core build time:
# concrete TRT runtimes live in backend DSOs.
set(_trtmc_default_search_paths
  /usr/include/x86_64-linux-gnu
  /usr/include/aarch64-linux-gnu
  /usr/include
  /usr/lib/x86_64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  /lib/x86_64-linux-gnu
  /lib/aarch64-linux-gnu
  /usr/local/cuda
  /usr/local/cuda/include
  /usr/local/cuda/lib64
  /usr/local/cuda/targets/sbsa-linux/include
  /usr/local/cuda/targets/sbsa-linux/lib
)

if(DEFINED ENV{TRT_ROOT})
  list(APPEND _trtmc_default_search_paths "$ENV{TRT_ROOT}")
endif()

if(DEFINED ENV{CUDA_HOME})
  list(APPEND _trtmc_default_search_paths "$ENV{CUDA_HOME}")
  list(APPEND _trtmc_default_search_paths "$ENV{CUDA_HOME}/include")
  list(APPEND _trtmc_default_search_paths "$ENV{CUDA_HOME}/lib64")
endif()

find_path(TRTMC_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS ${_trtmc_default_search_paths}
  PATH_SUFFIXES "" include include/zapped_headers
)

find_library(TRTMC_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so.11 nvinfer.so.11 libnvinfer.so.10 nvinfer.so.10
  PATHS ${_trtmc_default_search_paths}
  PATH_SUFFIXES "" lib lib64
)

find_path(TRTMC_CUDA_INCLUDE_DIR cuda_runtime_api.h
  PATHS ${_trtmc_default_search_paths}
  PATH_SUFFIXES "" include
)

find_library(TRTMC_CUDART_LIBRARY
  NAMES cudart libcudart.so
  PATHS ${_trtmc_default_search_paths}
  PATH_SUFFIXES "" lib lib64
)

function(_trtmc_header_define_value OUT_VAR DEFINE_NAME)
  set(${OUT_VAR} "" PARENT_SCOPE)
  if(NOT TRTMC_TRT_INCLUDE_DIR OR NOT EXISTS "${TRTMC_TRT_INCLUDE_DIR}/NvInferVersion.h")
    return()
  endif()
  file(READ "${TRTMC_TRT_INCLUDE_DIR}/NvInferVersion.h" _trtmc_version_header)
  string(REGEX MATCH "#define[ \t]+${DEFINE_NAME}[ \t]+([A-Za-z0-9_]+)"
    _trtmc_define_match "${_trtmc_version_header}")
  if(NOT _trtmc_define_match)
    return()
  endif()
  set(_trtmc_value "${CMAKE_MATCH_1}")
  if(NOT _trtmc_value MATCHES "^[0-9]+$")
    string(REGEX MATCH "#define[ \t]+${_trtmc_value}[ \t]+([0-9]+)"
      _trtmc_alias_match "${_trtmc_version_header}")
    if(_trtmc_alias_match)
      set(_trtmc_value "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(_trtmc_value MATCHES "^[0-9]+$")
    set(${OUT_VAR} "${_trtmc_value}" PARENT_SCOPE)
  endif()
endfunction()

if(NOT TRTMC_TRT_BACKEND_ABI AND TRTMC_TRT_INCLUDE_DIR)
  _trtmc_header_define_value(_trtmc_header_trt_major NV_TENSORRT_MAJOR)
  _trtmc_header_define_value(_trtmc_header_trt_minor NV_TENSORRT_MINOR)
  if(_trtmc_header_trt_major AND _trtmc_header_trt_minor)
    set(TRTMC_TRT_BACKEND_ABI "${_trtmc_header_trt_major}_${_trtmc_header_trt_minor}"
      CACHE STRING "TensorRT ABI suffix for the standard backend DSO" FORCE)
  endif()
endif()

if(NOT TRTMC_CUDA_INCLUDE_DIR OR NOT TRTMC_CUDART_LIBRARY)
  message(FATAL_ERROR
    "CUDA dev dependencies are required but not found.\n"
    "  TRTMC_CUDA_INCLUDE_DIR=${TRTMC_CUDA_INCLUDE_DIR}\n"
    "  TRTMC_CUDART_LIBRARY=${TRTMC_CUDART_LIBRARY}")
endif()

set(_trtmc_can_build_trt_backend FALSE)
if(TRTMC_ENABLE_TRT AND TRTMC_TRT_INCLUDE_DIR AND TRTMC_TRT_LIBRARY)
  set(_trtmc_can_build_trt_backend TRUE)
endif()

# --- nlohmann/json for BPE tokenizer JSON parsing ---
find_package(nlohmann_json CONFIG QUIET)
if(TARGET nlohmann_json::nlohmann_json)
  get_target_property(_trtmc_nlohmann_json_include
    nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "nlohmann/json: using package config")
else()
  include(FetchContent)
  set(JSON_Install OFF CACHE INTERNAL "")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
  set(_trtmc_nlohmann_json_include "${nlohmann_json_SOURCE_DIR}/include")
endif()

# --- Bundle-only runtime library ---
# PIC needed because trtmc_core is linked into backend DSOs (shared libs).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(cmake/trtmc_config_schemas.cmake)
include(cmake/trtmc_pipeline_plugins.cmake)
add_library(trtmc_core
  src/bundle/bundle_format.cpp
  src/bundle/bundle_view.cpp
  src/cabi/api/trtmc_c.cpp
  # Core runtime infrastructure (TRT engine, CUDA, KV cache, decode)
  src/runtime/core/trt_common.cpp
  src/runtime/core/gpu_matmul.cpp
  src/runtime/core/cuda_common.cpp
  src/runtime/backend/backend_loader.cpp
  src/runtime/backend/trt_version.cpp
  src/runtime/core/trt_engine_lifecycle.cpp
  src/runtime/core/distributed_runtime.cpp
  src/runtime/core/trt_decode_runtime.cpp
  src/runtime/core/sampler.cpp
  src/runtime/core/device_kv_cache.cpp
  src/runtime/core/device_tensor.cpp
  src/runtime/core/kv_cache.cpp
  src/runtime/core/triattention_kv_cache.cpp
  src/runtime/core/recurrent_state.cpp
  src/runtime/core/chat_template.cpp
  src/runtime/core/flow_match_euler_scheduler.cpp
  src/runtime/core/stb_impl.cpp
  # Domain-specific components
  src/runtime/domains/multimodal/image_preprocessor.cpp
  src/runtime/domains/multimodal/vision_engine.cpp
  src/runtime/domains/audio/audio_types.cpp
  src/runtime/domains/audio/mel_spectrogram.cpp
  src/runtime/domains/audio/audio_bundle_validation.cpp
  src/runtime/domains/diffusion/batch_utils.cpp
  src/runtime/domains/diffusion/diffusion_preprocessor.cpp
  src/runtime/domains/diffusion/qwen_image_types.cpp
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
  src/runtime/core/hybrid_state.cpp
  # Config (schema registry + layered resolution + CLI surface + schemas)
  src/runtime/config/schema_registry.cpp
  src/runtime/config/config_bundle.cpp
  src/runtime/config/cli_support.cpp
  ${TRTMC_CONFIG_SCHEMA_REGISTRATION_SOURCE}
  ${TRTMC_CONFIG_SCHEMA_SOURCES}
  # Registry (factory + plugin dispatch)
  src/runtime/registry/pipeline_factory.cpp
  src/runtime/registry/pipeline_registry.cpp
  src/runtime/registry/pipeline_plugin.cpp
  # Plugin shared helpers
  src/runtime/plugins/shared/plugin_helpers.cpp
  src/runtime/plugins/shared/diffusion_helpers.cpp
  src/runtime/plugins/shared/audio_helpers.cpp
  ${TRTMC_PIPELINE_PLUGIN_REGISTRATION_SOURCE}
  ${TRTMC_PIPELINE_PLUGIN_SOURCES}
  # Pipeline implementations
  src/runtime/models/text_generation/pipeline.cpp
  src/runtime/models/recurrent/pipeline.cpp
  src/runtime/models/encoder/pipeline.cpp
  src/runtime/models/patchtst/pipeline.cpp
  src/runtime/models/patchtsmixer/pipeline.cpp
  src/runtime/models/timesfm/pipeline.cpp
  src/runtime/models/chronos_bolt/pipeline.cpp
  src/runtime/models/elf_flow/pipeline.cpp
  src/runtime/models/segmentation/segment_pipeline.cpp
  src/runtime/models/segmentation/sam_pipeline.cpp
  src/runtime/models/timm_vit/pipeline.cpp
  src/runtime/models/vision_language/pipeline.cpp
  src/runtime/models/flux/pipeline.cpp
  src/runtime/models/ltx_video/pipeline.cpp
  src/runtime/models/wan/pipeline.cpp
  src/runtime/models/z_image/pipeline.cpp
  src/runtime/models/qwen_image/pipeline.cpp
  src/runtime/models/whisper/pipeline.cpp
  src/runtime/models/rnnt/pipeline.cpp
  src/runtime/models/bark/pipeline.cpp
  src/runtime/models/magpie/pipeline.cpp
  src/runtime/models/speech/pipeline.cpp
  src/runtime/models/omni/pipeline.cpp
  src/runtime/models/pixart/pipeline.cpp
  src/runtime/models/pixart_torchtrt/pipeline.cpp
)

# Keep warning policy strict for project code, but avoid CI log spam from
# vendored STB implementation headers and known GCC 13/libstdc++ false
# positives from inlined std::vector assignment helpers.
set_source_files_properties(src/runtime/core/stb_impl.cpp
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-pedantic>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-missing-field-initializers>"
)
set_source_files_properties(
  src/runtime/models/magpie/pipeline.cpp
  src/runtime/models/pixart/pipeline.cpp
  src/runtime/models/speech/pipeline.cpp
  src/runtime/models/wan/pipeline.cpp
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

# Add CUDA kernel sources when CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_core PRIVATE
    src/runtime/domains/audio/magpie_kernels.cu
    src/runtime/core/argmax_kernel.cu
    src/runtime/core/sparse_multinomial_kernel.cu
    src/runtime/core/triattention_kernels.cu
  )
  set_source_files_properties(
    src/runtime/domains/audio/magpie_kernels.cu
    src/runtime/core/argmax_kernel.cu
    src/runtime/core/sparse_multinomial_kernel.cu
    src/runtime/core/triattention_kernels.cu
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(trtmc_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(trtmc_core PUBLIC TRTMC_HAS_CUDA_KERNELS=1)
endif()

target_include_directories(trtmc_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(trtmc_core SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/stb
)

target_compile_definitions(trtmc_core
  PUBLIC
    TRTMC_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"
    TRTMC_VERSION_STRING=\"${PROJECT_VERSION}\"
)

target_compile_options(trtmc_core
  PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)

# nlohmann/json: use include-only (header-only lib) to avoid export-set issues
target_include_directories(trtmc_core SYSTEM PRIVATE
  ${_trtmc_nlohmann_json_include}
)

target_link_libraries(trtmc_core PRIVATE ${CMAKE_DL_LIBS})

target_compile_definitions(trtmc_core PRIVATE TRTMC_HAS_TRT=0)
target_include_directories(trtmc_core SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
find_library(TRTMC_CUBLAS_LIBRARY NAMES cublas libcublas.so
  PATHS ${_trtmc_default_search_paths} PATH_SUFFIXES "" lib lib64)
target_link_libraries(trtmc_core PRIVATE ${TRTMC_CUDART_LIBRARY}
  $<$<BOOL:${TRTMC_CUBLAS_LIBRARY}>:${TRTMC_CUBLAS_LIBRARY}>)
message(STATUS "Core runtime: TensorRT-free (backend DSOs are dlopen-loaded)")
message(STATUS "  CUDA include: ${TRTMC_CUDA_INCLUDE_DIR}")
message(STATUS "  CUDART lib : ${TRTMC_CUDART_LIBRARY}")
if(_trtmc_can_build_trt_backend)
  message(STATUS "  Standard TRT backend SDK detected: ${TRTMC_TRT_LIBRARY}")
else()
  message(STATUS "  Standard TRT backend SDK not detected; libtrtmc_backend_trt.so will be skipped")
endif()

# --- libtorch multinomial bridge (optional) ---
if(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL)
  set(_trtmc_torch_roots
    /opt/venv/lib/python3.12/site-packages/torch
    ${PROJECT_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch
  )
  if(DEFINED ENV{TRTMC_TORCH_ROOT})
    list(INSERT _trtmc_torch_roots 0 "$ENV{TRTMC_TORCH_ROOT}")
  endif()

  find_path(TRTMC_TORCH_INCLUDE_DIR ATen/ATen.h
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES include
  )
  find_path(TRTMC_TORCH_API_INCLUDE_DIR torch/torch.h
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES include/torch/csrc/api/include
  )
  find_library(TRTMC_TORCH_LIBRARY
    NAMES torch libtorch.so
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES lib
  )
  find_library(TRTMC_TORCH_CPU_LIBRARY
    NAMES torch_cpu libtorch_cpu.so
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES lib
  )
  find_library(TRTMC_TORCH_CUDA_LIBRARY
    NAMES torch_cuda libtorch_cuda.so
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES lib
  )
  find_library(TRTMC_C10_LIBRARY
    NAMES c10 libc10.so
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES lib
  )
  find_library(TRTMC_C10_CUDA_LIBRARY
    NAMES c10_cuda libc10_cuda.so
    PATHS ${_trtmc_torch_roots}
    PATH_SUFFIXES lib
  )
endif()

if(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL
   AND TRTMC_TORCH_INCLUDE_DIR AND TRTMC_TORCH_API_INCLUDE_DIR
   AND TRTMC_TORCH_LIBRARY AND TRTMC_TORCH_CPU_LIBRARY
   AND TRTMC_TORCH_CUDA_LIBRARY AND TRTMC_C10_LIBRARY AND TRTMC_C10_CUDA_LIBRARY)
  get_filename_component(TRTMC_TORCH_LIBRARY_DIR ${TRTMC_TORCH_LIBRARY} DIRECTORY)
  target_compile_definitions(trtmc_core PUBLIC TRTMC_HAS_LIBTORCH_MULTINOMIAL=1)
  target_include_directories(trtmc_core SYSTEM PRIVATE
    ${TRTMC_TORCH_INCLUDE_DIR}
    ${TRTMC_TORCH_API_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_core PRIVATE
    ${TRTMC_TORCH_LIBRARY}
    ${TRTMC_TORCH_CPU_LIBRARY}
    ${TRTMC_TORCH_CUDA_LIBRARY}
    ${TRTMC_C10_LIBRARY}
    ${TRTMC_C10_CUDA_LIBRARY}
  )
  set_property(TARGET trtmc_core APPEND PROPERTY BUILD_RPATH ${TRTMC_TORCH_LIBRARY_DIR})
  message(STATUS "libtorch multinomial bridge enabled")
  message(STATUS "  Torch include: ${TRTMC_TORCH_INCLUDE_DIR}")
  message(STATUS "  Torch API    : ${TRTMC_TORCH_API_INCLUDE_DIR}")
  message(STATUS "  Torch lib dir: ${TRTMC_TORCH_LIBRARY_DIR}")
else()
  target_compile_definitions(trtmc_core PUBLIC TRTMC_HAS_LIBTORCH_MULTINOMIAL=0)
  message(STATUS "libtorch multinomial bridge disabled")
endif()

# --- TVM-FFI kernel bridge plugin (optional) ---
option(TRTMC_ENABLE_TVM_FFI "Enable TVM-FFI kernel bridge plugin" ON)

set(_trtmc_has_tvm_ffi FALSE)

if(TRTMC_ENABLE_TVM_FFI)
  find_path(TRTMC_TVM_FFI_INCLUDE_DIR tvm/ffi/c_api.h
    PATHS ${_trtmc_default_search_paths}
    PATH_SUFFIXES "" include
  )
  find_library(TRTMC_TVM_FFI_LIBRARY
    NAMES tvm_ffi libtvm_ffi.so
    PATHS ${_trtmc_default_search_paths}
    PATH_SUFFIXES "" lib lib64
  )

  if(TRTMC_TVM_FFI_INCLUDE_DIR AND TRTMC_TVM_FFI_LIBRARY)
    set(_trtmc_has_tvm_ffi TRUE)
    target_sources(trtmc_core PRIVATE
      src/plugins/tvm_ffi_module_loader.cpp
    )
    target_compile_definitions(trtmc_core PUBLIC TRTMC_HAS_TVM_FFI=1)
    target_include_directories(trtmc_core SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtmc_core PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
    message(STATUS "TVM-FFI module loader enabled")
    message(STATUS "  TVM-FFI include: ${TRTMC_TVM_FFI_INCLUDE_DIR}")
    message(STATUS "  TVM-FFI library: ${TRTMC_TVM_FFI_LIBRARY}")
  endif()
endif()

if(NOT _trtmc_has_tvm_ffi)
  target_compile_definitions(trtmc_core PUBLIC TRTMC_HAS_TVM_FFI=0)
  message(STATUS "TVM-FFI kernel bridge disabled")
endif()

# Build the TVM-FFI TensorRT plugin as a standalone shared library.
if(_trtmc_has_tvm_ffi AND _trtmc_can_build_trt_backend)
  add_library(trtmc_tvm_ffi_plugin SHARED
    src/plugins/tvm_ffi_kernel_plugin.cpp
    src/plugins/tvm_ffi_kernel_creator.cpp
    src/plugins/tvm_ffi_module_loader.cpp
    src/utils/json_helpers.cpp
  )
  target_include_directories(trtmc_tvm_ffi_plugin PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(trtmc_tvm_ffi_plugin SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR} ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_tvm_ffi_plugin PRIVATE
    ${TRTMC_TRT_LIBRARY} ${TRTMC_CUDART_LIBRARY} ${TRTMC_TVM_FFI_LIBRARY})
  target_compile_definitions(trtmc_tvm_ffi_plugin PRIVATE TRTMC_HAS_TRT=1 TRTMC_HAS_TVM_FFI=1)
  target_compile_options(trtmc_tvm_ffi_plugin PRIVATE -Wall -Wextra -Wpedantic)
  message(STATUS "TVM-FFI plugin shared library: trtmc_tvm_ffi_plugin.so")
endif()

# --- Backend DSO: standard TRT ---
if(TRTMC_BUILD_BACKEND_TRT AND _trtmc_can_build_trt_backend)
  add_library(trtmc_backend_trt SHARED
    src/runtime/backend/trt_logger.cpp
    src/runtime/backend/trt_module_impl.cpp
    src/runtime/backend/trt_backend.cpp
  )
  target_include_directories(trtmc_backend_trt
    PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
  )
  target_include_directories(trtmc_backend_trt SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_backend_trt PRIVATE
    trtmc_core ${TRTMC_TRT_LIBRARY} ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS}
  )
  if(_trtmc_has_tvm_ffi)
    target_sources(trtmc_backend_trt PRIVATE
      src/plugins/tvm_ffi_kernel_plugin.cpp
      src/plugins/tvm_ffi_kernel_creator.cpp
    )
    target_include_directories(trtmc_backend_trt SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtmc_backend_trt PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
  endif()
  target_compile_options(trtmc_backend_trt PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(trtmc_backend_trt PRIVATE
    TRTMC_HAS_TRT=1
    TRTMC_HAS_TVM_FFI=$<BOOL:${_trtmc_has_tvm_ffi}>
    TRTMC_TRT_BACKEND_ABI_STRING=\"${TRTMC_TRT_BACKEND_ABI}\"
  )
  set_target_properties(trtmc_backend_trt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    SKIP_BUILD_RPATH TRUE
    INSTALL_RPATH ""
  )
  if(TRTMC_TRT_BACKEND_ABI)
    add_custom_command(TARGET trtmc_backend_trt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:trtmc_backend_trt>
        "${CMAKE_BINARY_DIR}/libtrtmc_backend_trt_${TRTMC_TRT_BACKEND_ABI}.so"
      COMMENT "Writing TensorRT ABI backend alias libtrtmc_backend_trt_${TRTMC_TRT_BACKEND_ABI}.so"
    )
  endif()
  # Also add nlohmann_json include for trt_module_impl (uses DType from tensor.h)
  target_include_directories(trtmc_backend_trt SYSTEM PRIVATE
    ${_trtmc_nlohmann_json_include}
  )
  message(STATUS "TRT backend DSO: enabled (libtrtmc_backend_trt.so)")
  if(TRTMC_TRT_BACKEND_ABI)
    message(STATUS "TRT backend DSO ABI alias: libtrtmc_backend_trt_${TRTMC_TRT_BACKEND_ABI}.so")
  endif()
elseif(TRTMC_BUILD_BACKEND_TRT)
  message(STATUS "TRT backend DSO: disabled (missing libnvinfer or headers)")
endif()

# --- Backend DSO: TRT-RTX (optional) ---
option(TRTMC_BUILD_BACKEND_RTX "Build TRT-RTX backend DSO" OFF)
if(TRTMC_BUILD_BACKEND_RTX)
  find_library(TENSORRT_RTX_LIB tensorrt_rtx HINTS ${TRTMC_RTX_LIBRARY_DIR})
  # Use TRTMC_RTX_INCLUDE_DIR directly; find_path can pick up standard TRT first.
  set(TRT_RTX_INCLUDE_DIR "${TRTMC_RTX_INCLUDE_DIR}")
  if(TENSORRT_RTX_LIB AND TRT_RTX_INCLUDE_DIR)
    add_library(trtmc_backend_rtx SHARED
      src/runtime/backend/trt_logger.cpp
      src/runtime/backend/trt_module_impl.cpp
      src/runtime/backend/rtx_backend.cpp
    )
    target_include_directories(trtmc_backend_rtx
      PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )
    target_include_directories(trtmc_backend_rtx SYSTEM PRIVATE
      ${TRT_RTX_INCLUDE_DIR}
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(trtmc_backend_rtx PRIVATE
      trtmc_core ${TENSORRT_RTX_LIB} ${TRTMC_CUDART_LIBRARY}
    )
    if(_trtmc_has_tvm_ffi)
      target_sources(trtmc_backend_rtx PRIVATE
        src/plugins/tvm_ffi_kernel_plugin.cpp
        src/plugins/tvm_ffi_kernel_creator.cpp
      )
      target_include_directories(trtmc_backend_rtx SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
      target_link_libraries(trtmc_backend_rtx PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
    endif()
    target_compile_options(trtmc_backend_rtx PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(trtmc_backend_rtx PRIVATE
      TRTMC_HAS_TRT=1
      TRTMC_HAS_TVM_FFI=$<BOOL:${_trtmc_has_tvm_ffi}>
    )
    set_target_properties(trtmc_backend_rtx PROPERTIES
      OUTPUT_NAME trtmc_backend_trt_rtx
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_include_directories(trtmc_backend_rtx SYSTEM PRIVATE
      ${_trtmc_nlohmann_json_include}
    )
    message(STATUS "TRT-RTX backend DSO: enabled (libtrtmc_backend_trt_rtx.so)")
  else()
    message(STATUS "TRT-RTX backend DSO: disabled (missing tensorrt_rtx lib or headers)")
  endif()
endif()

add_executable(trtmc
  src/cli/args.cpp
  src/cli/main.cpp
)
target_link_libraries(trtmc PRIVATE trtmc_core)
target_include_directories(trtmc PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(trtmc SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)

if(TRTMC_BUILD_BENCHMARKS)
  add_executable(trtmc_dataset_benchmark examples/trtmc_dataset_benchmark.cpp)
  target_link_libraries(trtmc_dataset_benchmark PRIVATE trtmc_core)
  target_include_directories(trtmc_dataset_benchmark SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
endif()

if(TRTMC_BUILD_TESTS)
enable_testing()
add_custom_target(trtmc_cpp_tests)

# --- Test helper function ---
# Usage:
#   trtmc_add_test(test_name)               -- links trtmc_core, adds src/ include
#   trtmc_add_test(test_name REQUIRES_TRT)  -- also links TensorRT test deps
function(trtmc_add_test TEST_NAME)
  cmake_parse_arguments(ARG "NO_SRC_INCLUDE;REQUIRES_TRT" "" "EXTRA_INCLUDES" ${ARGN})
  if(ARG_REQUIRES_TRT AND NOT _trtmc_can_build_trt_backend)
    message(STATUS "Skipping ${TEST_NAME}: TensorRT SDK not available")
    return()
  endif()
  add_executable(${TEST_NAME} EXCLUDE_FROM_ALL tests/cpp/${TEST_NAME}.cpp)
  add_dependencies(trtmc_cpp_tests ${TEST_NAME})
  target_link_libraries(${TEST_NAME} PRIVATE trtmc_core)
  if(NOT ARG_NO_SRC_INCLUDE)
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
  endif()
  if(ARG_EXTRA_INCLUDES)
    target_include_directories(${TEST_NAME} PRIVATE ${ARG_EXTRA_INCLUDES})
  endif()
  target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(${TEST_NAME} PRIVATE ${TRTMC_CUDART_LIBRARY})
  if(ARG_REQUIRES_TRT)
    target_compile_definitions(${TEST_NAME} PRIVATE TRTMC_HAS_TRT=1)
    target_include_directories(${TEST_NAME} SYSTEM PRIVATE ${TRTMC_TRT_INCLUDE_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE ${TRTMC_TRT_LIBRARY})
  else()
    target_compile_definitions(${TEST_NAME} PRIVATE TRTMC_HAS_TRT=0)
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Tests ---

# Bundle / format
trtmc_add_test(test_bundle_format)
trtmc_add_test(test_bundle_e2e          NO_SRC_INCLUDE)
trtmc_add_test(test_bundle_view)
trtmc_add_test(test_pipeline_registry)
trtmc_add_test(test_backend_loader)
trtmc_add_test(test_trt_version)

# C ABI
trtmc_add_test(test_c_abi_entry                NO_SRC_INCLUDE)
trtmc_add_test(test_c_abi_runtime_regression   NO_SRC_INCLUDE)

# Config / parsing
trtmc_add_test(test_text_parsers)
trtmc_add_test(test_json_helpers)
trtmc_add_test(test_data_dir)
trtmc_add_test(test_cli_args)
target_sources(test_cli_args PRIVATE src/cli/args.cpp)
trtmc_add_test(test_chat_template)
trtmc_add_test(test_config_schema_registry)
trtmc_add_test(test_config_cli_support)
trtmc_add_test(test_config_schemas_triattention)

# TRT runtime core
trtmc_add_test(test_decode_runtime REQUIRES_TRT)
trtmc_add_test(test_sampler)
trtmc_add_test(test_plugin_helpers)
trtmc_add_test(test_trt_logger REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_logger PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_trt_runtime_lifetime REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_runtime_lifetime PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_trt_engine_lifecycle)
trtmc_add_test(test_trt_engine_lifecycle_fake_engine)
trtmc_add_test(test_trt_module REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_module PRIVATE trtmc_backend_trt)
endif()

# CUDA / device
trtmc_add_test(test_cuda_buffer REQUIRES_TRT)
trtmc_add_test(test_cuda_stream REQUIRES_TRT)
trtmc_add_test(test_cuda_graph REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_cuda_graph PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_device_kv_cache)
trtmc_add_test(test_device_resources REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_device_resources PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_device_tensor)
trtmc_add_test(test_kv_cache_new)
trtmc_add_test(test_triattention_kv_cache)
trtmc_add_test(test_recurrent_state)

# Pipelines (public API)
trtmc_add_test(test_pipeline_api                NO_SRC_INCLUDE)
trtmc_add_test(test_text_generation_pipeline REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_text_generation_pipeline PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_elf_flow_pipeline)
trtmc_add_test(test_recurrent_pipeline REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_recurrent_pipeline PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_encoder_pipeline REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_encoder_pipeline PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_vl_pipeline REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_vl_pipeline PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_diffusion_pipeline_new)
trtmc_add_test(test_audio_pipeline_new REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_audio_pipeline_new PRIVATE trtmc_backend_trt)
endif()

# Tokenizers
trtmc_add_test(test_vocab_tokenizer     NO_SRC_INCLUDE)
trtmc_add_test(test_ipa_tokenizer)
trtmc_add_test(test_bpe_tokenizer)
trtmc_add_test(test_bpe_golden)
trtmc_add_test(test_bpe_benchmark)
trtmc_add_test(test_wordpiece_tokenizer)
trtmc_add_test(test_wordpiece_golden)
trtmc_add_test(test_unigram_tokenizer)
trtmc_add_test(test_unigram_golden)
trtmc_add_test(test_tokenizer_edge_cases)

# Image I/O
trtmc_add_test(test_image_reader        NO_SRC_INCLUDE)
trtmc_add_test(test_trtmc_io             NO_SRC_INCLUDE)

# Vision / VL
trtmc_add_test(test_image_preprocessor)
trtmc_add_test(test_vl_decode_policy)
trtmc_add_test(test_vision_execution_plan)

# Diffusion
trtmc_add_test(test_diffusion_generation_plan)
trtmc_add_test(test_diffusion_denoising_step_seam)
trtmc_add_test(test_flow_match_scheduler)
trtmc_add_test(test_diffusion_math)
trtmc_add_test(test_diffusion_batch_utils)
# PR 2 of diffusion batch-inference: bundle-free combiner-level unit that
# proves combine_cfg_with_renorm reduces per-token (no cross-sample leak).
# Full batch-vs-sequential parity lands as E2E tests in PR 3.
trtmc_add_test(test_qwen_image_cfg_renorm)
trtmc_add_test(test_ltx_video_pipeline)

# Recurrent (Mamba / RWKV)
trtmc_add_test(test_recurrent_step_contracts)

# Audio — Whisper
trtmc_add_test(test_audio_bundle_validation)
trtmc_add_test(test_whisper_host_plan)
trtmc_add_test(test_whisper_decode_policy)
trtmc_add_test(test_rnnt_decode_policy)
trtmc_add_test(test_rnnt_streaming_contract)
trtmc_add_test(test_wav_reader)
trtmc_add_test(test_audio_types)
trtmc_add_test(test_mel_spectrogram)

# Audio — Bark
trtmc_add_test(test_bark_generation_plan)

# Audio — Speech (PersonaPlex / Magpie)
trtmc_add_test(test_speech_subprocess_seam      NO_SRC_INCLUDE)
trtmc_add_test(test_speech_decode_stop_policy   NO_SRC_INCLUDE)
trtmc_add_test(test_speech_generation_helpers)
trtmc_add_test(test_speech_depth_plan)
trtmc_add_test(test_speech_runtime_plan)
trtmc_add_test(test_speech_temporal_embed_plan)
trtmc_add_test(test_speech_mimi_decode_plan)
trtmc_add_test(test_magpie_decode_policy)
trtmc_add_test(test_magpie_decoder_plan)
trtmc_add_test(test_magpie_text_completion_policy)
trtmc_add_test(test_magpie_codec_plan)

# Audio — Omni
trtmc_add_test(test_omni_audio_plan)

# Perception (segmentation, SAM, neural operator)
trtmc_add_test(test_neural_operator_config)
trtmc_add_test(test_sam_prompt_seam)
trtmc_add_test(test_perception_preprocess_seams)

# Diffusion — conditioning
trtmc_add_test(test_wan_generation_conditioning)

# IoMap tensor name mapping
trtmc_add_test(test_io_map)

# Decoder config parsing (legacy name: test_torchtrt_decoder)
trtmc_add_test(test_torchtrt_decoder)

# TVM-FFI plugin
trtmc_add_test(test_tvm_ffi_plugin REQUIRES_TRT)
if(TARGET test_tvm_ffi_plugin AND _trtmc_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
endif()

# TVM-FFI plugin V2 (self-contained test with IPluginV2DynamicExt)
trtmc_add_test(test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE REQUIRES_TRT)

# TVM-FFI module loader (loads FlashInfer .so from C++)
trtmc_add_test(test_tvm_ffi_module_loader REQUIRES_TRT)
if(TARGET test_tvm_ffi_module_loader AND _trtmc_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_module_loader SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_module_loader PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
endif()
if(TARGET test_tvm_ffi_plugin_v2 AND _trtmc_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin_v2 SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin_v2 PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
endif()
endif()

# --- Install ---
include(CMakePackageConfigHelpers)

# Install library
install(TARGETS trtmc_core
  EXPORT trtmcTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET trtmc_backend_trt)
  install(TARGETS trtmc_backend_trt
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  if(TRTMC_TRT_BACKEND_ABI)
    install(FILES "${CMAKE_BINARY_DIR}/libtrtmc_backend_trt_${TRTMC_TRT_BACKEND_ABI}.so"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      OPTIONAL
    )
  endif()
endif()

# Install public headers
install(DIRECTORY include/trtmc/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install CLI executable
install(TARGETS trtmc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Generate and install CMake config files
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/trtmcConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT trtmcTargets
  FILE trtmcTargets.cmake
  NAMESPACE trtmc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)

install(FILES
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)
