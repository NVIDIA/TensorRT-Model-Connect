# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(tensorrt_model_connect VERSION 0.1.0 LANGUAGES CXX)

# Release wheels package build outputs directly. Keep CMake's automatically
# generated build-tree RPATH entries relative to each ELF file.
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Standard CUDA installations do not always add their bin directory to PATH
# (notably bare-metal L4T images). Preserve an explicit CUDACXX selection, but
# otherwise probe the same CUDA roots used later for headers and libraries.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  find_program(_trtmc_cuda_compiler
    NAMES nvcc
    HINTS
      "$ENV{CUDA_HOME}"
      "$ENV{CUDAToolkit_ROOT}"
      "$ENV{CUDA_PATH}"
    PATHS
      /usr/local/cuda
      /opt/cuda
    PATH_SUFFIXES bin
  )
  if(_trtmc_cuda_compiler)
    set(CMAKE_CUDA_COMPILER "${_trtmc_cuda_compiler}"
      CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

# Enable CUDA for runtime GPU kernels when a CUDA compiler is available.
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
find_package(Threads REQUIRED)

option(TRTMC_ENABLE_TRT "Enable TensorRT backend DSO integration" ON)
option(TRTMC_BUILD_BACKEND_TRT "Build standard TensorRT backend DSO" ON)
option(TRTMC_BUILD_TESTS "Build C++ unit test executables" ON)
option(TRTMC_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL "Enable optional libtorch-backed multinomial sampler" ON)
option(TRTMC_DISTRIBUTABLE_BUILD
       "Build distributable artifacts without embedding source/build paths" OFF)
set(TRTMC_MODEL_PROOF_MODEL "" CACHE STRING
  "Require the source tree to contain exactly this runtime model (CI isolation proof mode).")
set(TRTMC_TRT_BACKEND_ABI "" CACHE STRING
  "TensorRT ABI suffix for the standard backend DSO, e.g. 10_15. Empty means auto-detect from headers.")
set(TRTMC_SOURCE_REVISION "" CACHE STRING
  "Source revision embedded in release benchmark worker provenance.")
if(NOT TRTMC_SOURCE_REVISION)
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE _trtmc_source_revision_status
    OUTPUT_VARIABLE _trtmc_source_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_trtmc_source_revision_status EQUAL 0)
    set(TRTMC_SOURCE_REVISION "${_trtmc_source_revision}")
  else()
    set(TRTMC_SOURCE_REVISION "unknown")
  endif()
endif()

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

# The TensorRT pip wheel (tensorrt_cu13) ships libnvinfer.so under a Python
# site-packages tensorrt_libs/ tree. Ask Python where so CMake can find it
# without manual hints. Mirrors the apache-tvm-ffi discovery below.
find_program(_trtmc_python NAMES python3.12 python3 python)
if(_trtmc_python)
  execute_process(
    COMMAND "${_trtmc_python}" -c
      "import importlib.util,sys; s=importlib.util.find_spec('tensorrt_libs'); sys.stdout.write(s.submodule_search_locations[0] if s else '')"
    OUTPUT_VARIABLE _trtmc_tensorrt_libs_pkg
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_trtmc_tensorrt_libs_pkg)
    list(APPEND _trtmc_default_search_paths "${_trtmc_tensorrt_libs_pkg}")
  endif()
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
add_library(trtmc_core SHARED
  src/bundle/bundle_format.cpp
  src/bundle/bundle_view.cpp
  src/cabi/api/trtmc_c.cpp
  # Core runtime infrastructure (TRT engine, CUDA, KV cache, decode)
  src/runtime/core/trt_common.cpp
  src/runtime/core/cuda_common.cpp
  src/runtime/backend/backend_loader.cpp
  src/runtime/backend/trt_version.cpp
  src/runtime/providers/optimized_runtime_host.cpp
  src/runtime/core/distributed_runtime.cpp
  src/runtime/core/device_tensor.cpp
  src/runtime/core/image_features.cpp
  src/runtime/core/speech_session.cpp
  src/runtime/core/measurement.cpp
  src/runtime/core/pipeline_pool.cpp
  src/runtime/core/stb_impl.cpp
  src/tokenizer/vocab_tokenizer.cpp
  src/tokenizer/ipa_tokenizer.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/tokenizer/wordpiece_tokenizer.cpp
  src/tokenizer/unigram_tokenizer.cpp
  # Utilities
  src/utils/data_dir.cpp
  src/utils/text_parsers.cpp
  src/utils/json_helpers.cpp
  src/cli/jsonl_io.cpp
  src/utils/wav_reader.cpp
  src/utils/image_reader.cpp
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
  src/runtime/registry/pipeline_plugin_loader.cpp
  ${TRTMC_MODEL_PLUGIN_INDEX_SOURCE}
)

# Keep warning policy strict for project code, but avoid CI log spam from
# vendored STB implementation headers and known GCC 13/libstdc++ false
# positives from inlined std::vector assignment helpers.
set_source_files_properties(src/runtime/core/stb_impl.cpp
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-pedantic>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-missing-field-initializers>"
)
# Add CUDA kernel sources when CUDA compiler is available
if(CMAKE_CUDA_COMPILER)
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

target_compile_definitions(trtmc_core PUBLIC TRTMC_VERSION_STRING=\"${PROJECT_VERSION}\")
if(NOT TRTMC_DISTRIBUTABLE_BUILD)
  target_compile_definitions(trtmc_core
    PUBLIC TRTMC_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"
    PRIVATE TRTMC_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
  )
endif()

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
  # apache-tvm-ffi is distributed as a pip wheel that ships its headers + .so
  # under the Python site-packages tvm_ffi/{include,lib}/ tree. Ask Python
  # where that is so cmake can find it without manual TRTMC_TVM_FFI_*_DIR.
  if(_trtmc_python)
    execute_process(
      COMMAND "${_trtmc_python}" -c
        "import importlib.util; s=importlib.util.find_spec('tvm_ffi'); import sys; sys.stdout.write(s.submodule_search_locations[0] if s else '')"
      OUTPUT_VARIABLE _trtmc_tvm_ffi_pkg
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_trtmc_tvm_ffi_pkg)
      list(APPEND _trtmc_default_search_paths
           "${_trtmc_tvm_ffi_pkg}/include"
           "${_trtmc_tvm_ffi_pkg}/lib")
    endif()
  endif()

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
      src/plugins/tvm_ffi_runtime_bindings.cpp
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

# --- Per-model runtime plugin DSOs ---
add_custom_target(trtmc_model_plugins)
foreach(_trtmc_model IN LISTS TRTMC_RUNTIME_MODEL_IDS)
  string(MAKE_C_IDENTIFIER "${_trtmc_model}" _trtmc_model_var)
  file(GLOB _trtmc_model_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/runtime/models/${_trtmc_model}/*.cpp")
  if(CMAKE_CUDA_COMPILER)
    file(GLOB _trtmc_model_cuda_sources CONFIGURE_DEPENDS
      "${PROJECT_SOURCE_DIR}/src/runtime/models/${_trtmc_model}/*.cu")
    if(_trtmc_model_cuda_sources)
      set_source_files_properties(${_trtmc_model_cuda_sources} PROPERTIES LANGUAGE CUDA)
    endif()
  else()
    set(_trtmc_model_cuda_sources)
  endif()
  foreach(_trtmc_suppressed_source
      IN LISTS TRTMC_MODEL_${_trtmc_model_var}_GNU_WARNING_SUPPRESSED_SOURCES)
    set_source_files_properties(
      "${PROJECT_SOURCE_DIR}/src/runtime/models/${_trtmc_model}/${_trtmc_suppressed_source}"
      PROPERTIES
        COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
    )
  endforeach()
  add_library(trtmc_model_${_trtmc_model} SHARED
    ${_trtmc_model_sources}
    ${_trtmc_model_cuda_sources}
    ${TRTMC_MODEL_${_trtmc_model_var}_REGISTRATION_SOURCE}
  )
  add_dependencies(trtmc_model_plugins trtmc_model_${_trtmc_model})
  target_include_directories(trtmc_model_${_trtmc_model}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/include
  )
  target_include_directories(trtmc_model_${_trtmc_model} SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
    ${_trtmc_nlohmann_json_include}
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(trtmc_model_${_trtmc_model} PRIVATE
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  foreach(_trtmc_model_link_library IN LISTS TRTMC_MODEL_${_trtmc_model_var}_LINK_LIBRARIES)
    if(_trtmc_model_link_library STREQUAL "cublas")
      if(NOT TRTMC_CUBLAS_LIBRARY)
        message(FATAL_ERROR
          "Model '${_trtmc_model}' requires runtime_link_libraries=['cublas'], "
          "but cuBLAS was not found")
      endif()
      target_link_libraries(trtmc_model_${_trtmc_model} PRIVATE ${TRTMC_CUBLAS_LIBRARY})
    else()
      message(FATAL_ERROR
        "Unknown runtime_link_libraries entry '${_trtmc_model_link_library}' "
        "for model '${_trtmc_model}'")
    endif()
  endforeach()
  target_compile_definitions(trtmc_model_${_trtmc_model} PRIVATE TRTMC_HAS_TRT=0)
  target_compile_options(trtmc_model_${_trtmc_model}
    PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
  )
  set_target_properties(trtmc_model_${_trtmc_model} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/${_trtmc_model}"
    BUILD_RPATH "\$ORIGIN/../.."
    INSTALL_RPATH "\$ORIGIN/../.."
  )
  install(TARGETS trtmc_model_${_trtmc_model}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/trtmc/models/${_trtmc_model}
  )
endforeach()

# Build the TVM-FFI TensorRT plugin as a standalone shared library.
if(_trtmc_has_tvm_ffi AND _trtmc_can_build_trt_backend)
  add_library(trtmc_tvm_ffi_plugin SHARED
    src/plugins/tvm_ffi_kernel_plugin.cpp
    src/plugins/tvm_ffi_kernel_creator.cpp
  )
  target_include_directories(trtmc_tvm_ffi_plugin PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(trtmc_tvm_ffi_plugin SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR} ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_tvm_ffi_plugin PRIVATE
    trtmc_core ${TRTMC_TRT_LIBRARY} ${TRTMC_CUDART_LIBRARY} ${TRTMC_TVM_FFI_LIBRARY})
  target_compile_definitions(trtmc_tvm_ffi_plugin PRIVATE TRTMC_HAS_TRT=1 TRTMC_HAS_TVM_FFI=1)
  target_compile_options(trtmc_tvm_ffi_plugin PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(trtmc_tvm_ffi_plugin PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "\$ORIGIN"
  )

  add_library(trtmc_byok_identity_copy SHARED EXCLUDE_FROM_ALL
    examples/byok/identity_copy_kernel.cpp
  )
  target_include_directories(trtmc_byok_identity_copy SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_byok_identity_copy PRIVATE
    ${TRTMC_CUDART_LIBRARY} ${TRTMC_TVM_FFI_LIBRARY})
  set_target_properties(trtmc_byok_identity_copy PROPERTIES
    PREFIX ""
    OUTPUT_NAME "identity_copy_kernel"
  )

  message(STATUS "TVM-FFI plugin shared library: trtmc_tvm_ffi_plugin.so")
endif()

# --- Optional: vendored diffusion CUDA kernels as TVM-FFI modules ---
# Off by default. Requires TVM-FFI. Each kernel builds as a standalone .so
# that the diffusion TRT graph loads via graph_ops.add_tvm_ffi_kernel().
option(TRTMC_BUILD_DIFFUSION_KERNELS
       "Build vendored optimized diffusion CUDA kernels (TVM-FFI modules)" OFF)

if(TRTMC_BUILD_DIFFUSION_KERNELS)
  if(NOT _trtmc_has_tvm_ffi)
    message(FATAL_ERROR
      "TRTMC_BUILD_DIFFUSION_KERNELS=ON requires TVM-FFI (TRTMC_ENABLE_TVM_FFI=ON, "
      "and a discoverable libtvm_ffi). Found none.")
  endif()
  add_subdirectory(src/runtime/domains/diffusion/kernels)
endif()

# --- Backend DSO: standard TRT ---
if(TRTMC_BUILD_BACKEND_TRT AND _trtmc_can_build_trt_backend)
  add_library(trtmc_backend_trt SHARED
    src/runtime/backend/trt_logger.cpp
    src/runtime/backend/trt_module_impl.cpp
    src/runtime/backend/trt_backend.cpp
  )
  if(CMAKE_CUDA_COMPILER)
    target_sources(trtmc_backend_trt PRIVATE
      src/plugins/top4_logits_refinement_plugin.cpp
      src/plugins/top4_logits_refinement_kernel.cu
    )
  endif()
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
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
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
target_link_libraries(trtmc PRIVATE trtmc_core Threads::Threads)
target_include_directories(trtmc PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(trtmc SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/stb
  ${_trtmc_nlohmann_json_include}
)
set_target_properties(trtmc PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH "\$ORIGIN"
)

if(TRTMC_BUILD_BENCHMARKS)
  add_executable(trtmc_dataset_benchmark
    examples/trtmc_dataset_benchmark.cpp
    src/cli/args.cpp
  )
  target_link_libraries(trtmc_dataset_benchmark PRIVATE trtmc_core)
  target_include_directories(trtmc_dataset_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(trtmc_dataset_benchmark SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
    ${_trtmc_nlohmann_json_include}
  )

  add_executable(trtmc_benchmark_worker examples/trtmc_benchmark_worker.cpp)
  target_link_libraries(trtmc_benchmark_worker PRIVATE trtmc_core)
  target_include_directories(trtmc_benchmark_worker SYSTEM PRIVATE
    ${_trtmc_nlohmann_json_include}
  )
  target_compile_definitions(trtmc_benchmark_worker PRIVATE
    TRTMC_BUILD_CONFIGURATION=\"$<CONFIG>\"
    TRTMC_SOURCE_REVISION=\"${TRTMC_SOURCE_REVISION}\"
  )
  set_target_properties(trtmc_benchmark_worker PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "\$ORIGIN"
  )
endif()

if(TRTMC_BUILD_TESTS)
enable_testing()
add_custom_target(trtmc_cpp_tests)
add_custom_target(trtmc_platform_cpp_tests)
add_custom_target(trtmc_cpu_cpp_tests)

# Generic capsule test doubles. The good and wrong-identity DSOs intentionally
# have the same filename in different directories so tests can prove that the
# embedded DSO's declared identity is validated before its factory is called.
add_library(trtmc_test_optimized_runtime SHARED EXCLUDE_FROM_ALL
  tests/cpp/fakes/fake_optimized_runtime_provider.cpp
)
target_include_directories(trtmc_test_optimized_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(trtmc_test_optimized_runtime PROPERTIES
  OUTPUT_NAME trtmc_impl_example_optimized_runtime
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/optimized-runtimes/good"
)

add_library(trtmc_test_wrong_optimized_runtime SHARED EXCLUDE_FROM_ALL
  tests/cpp/fakes/fake_optimized_runtime_provider.cpp
)
target_include_directories(trtmc_test_wrong_optimized_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(trtmc_test_wrong_optimized_runtime PRIVATE
  TRTMC_FAKE_OPTIMIZED_IMPLEMENTATION_ID="wrong-example-runtime"
)
set_target_properties(trtmc_test_wrong_optimized_runtime PROPERTIES
  OUTPUT_NAME trtmc_impl_example_optimized_runtime
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/optimized-runtimes/wrong"
)

add_library(trtmc_test_wrong_optimized_runtime_abi SHARED EXCLUDE_FROM_ALL
  tests/cpp/fakes/fake_optimized_runtime_provider.cpp
)
target_include_directories(trtmc_test_wrong_optimized_runtime_abi PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(trtmc_test_wrong_optimized_runtime_abi PRIVATE
  TRTMC_FAKE_OPTIMIZED_PIPELINE_ABI_VERSION=0
)
set_target_properties(trtmc_test_wrong_optimized_runtime_abi PROPERTIES
  OUTPUT_NAME trtmc_impl_example_optimized_runtime
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/optimized-runtimes/wrong-abi"
)

add_library(trtmc_test_optimized_embedding SHARED EXCLUDE_FROM_ALL
  tests/cpp/fakes/fake_optimized_runtime_embedding.cpp
)
target_include_directories(trtmc_test_optimized_embedding PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
set_target_properties(trtmc_test_optimized_embedding PROPERTIES
  OUTPUT_NAME trtmc_impl_example_optimized_embedding
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/optimized-runtimes/embedding"
)

# --- Test helper function ---
# Usage:
#   trtmc_add_test(test_name)               -- links trtmc_core, adds src/ include
#   trtmc_add_test(test_name REQUIRES_TRT)  -- also links TensorRT test deps
#   trtmc_add_test(test_name MODEL_OWNED)    -- labels model ownership independently
#   trtmc_add_test(test_name REQUIRES_GPU)   -- excludes it from all CPU unit targets
function(trtmc_add_test TEST_NAME)
  cmake_parse_arguments(
    ARG
    "NO_SRC_INCLUDE;REQUIRES_TRT;REQUIRES_GPU;MODEL_OWNED"
    "SOURCE"
    "EXTRA_INCLUDES"
    ${ARGN}
  )
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown trtmc_add_test option(s) for '${TEST_NAME}': ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_REQUIRES_TRT AND NOT _trtmc_can_build_trt_backend)
    message(STATUS "Skipping ${TEST_NAME}: TensorRT SDK not available")
    return()
  endif()
  if(ARG_SOURCE)
    if(IS_ABSOLUTE "${ARG_SOURCE}")
      set(_trtmc_test_source "${ARG_SOURCE}")
    else()
      set(_trtmc_test_source "${PROJECT_SOURCE_DIR}/${ARG_SOURCE}")
    endif()
  else()
    set(_trtmc_test_source "${PROJECT_SOURCE_DIR}/tests/cpp/${TEST_NAME}.cpp")
  endif()
  add_executable(${TEST_NAME} EXCLUDE_FROM_ALL "${_trtmc_test_source}")
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
    if(TARGET trtmc_backend_trt)
      target_link_libraries(${TEST_NAME} PRIVATE trtmc_backend_trt)
    endif()
  else()
    target_compile_definitions(${TEST_NAME} PRIVATE TRTMC_HAS_TRT=0)
  endif()
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  if(ARG_MODEL_OWNED)
    set(_trtmc_test_owner_label "model")
  else()
    set(_trtmc_test_owner_label "platform")
  endif()
  if(ARG_REQUIRES_GPU)
    set(_trtmc_test_resource_label "gpu")
  else()
    set(_trtmc_test_resource_label "cpu")
    add_dependencies(trtmc_cpu_cpp_tests ${TEST_NAME})
  endif()
  if(NOT ARG_MODEL_OWNED AND NOT ARG_REQUIRES_GPU)
    add_dependencies(trtmc_platform_cpp_tests ${TEST_NAME})
  endif()
  set_tests_properties(
    ${TEST_NAME}
    PROPERTIES LABELS "${_trtmc_test_owner_label};${_trtmc_test_resource_label}"
  )
endfunction()

function(_trtmc_manifest_csv_to_list VALUE OUTPUT_VAR)
  if(NOT VALUE OR VALUE STREQUAL "_")
    set(${OUTPUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  string(REPLACE "," ";" _trtmc_values "${VALUE}")
  set(${OUTPUT_VAR} ${_trtmc_values} PARENT_SCOPE)
endfunction()

function(trtmc_add_model_manifest_tests)
  foreach(_trtmc_model IN LISTS TRTMC_RUNTIME_MODEL_IDS)
    string(MAKE_C_IDENTIFIER "${_trtmc_model}" _trtmc_model_var)
    foreach(_trtmc_test_entry IN LISTS TRTMC_MODEL_${_trtmc_model_var}_TESTS)
      string(REPLACE "|" ";" _trtmc_test_fields "${_trtmc_test_entry}")
      list(LENGTH _trtmc_test_fields _trtmc_test_field_count)
      if(NOT _trtmc_test_field_count EQUAL 5)
        message(FATAL_ERROR
          "Invalid runtime_tests entry '${_trtmc_test_entry}' for model '${_trtmc_model}'; "
          "expected name|source|link_targets|extra_sources|options")
      endif()
      list(GET _trtmc_test_fields 0 _trtmc_test_name)
      list(GET _trtmc_test_fields 1 _trtmc_test_source)
      list(GET _trtmc_test_fields 2 _trtmc_link_targets_csv)
      list(GET _trtmc_test_fields 3 _trtmc_extra_sources_csv)
      list(GET _trtmc_test_fields 4 _trtmc_options_csv)

      _trtmc_manifest_csv_to_list("${_trtmc_options_csv}" _trtmc_test_options)
      trtmc_add_test(
        ${_trtmc_test_name}
        SOURCE "tests/cpp/models/${_trtmc_model}/${_trtmc_test_source}"
        MODEL_OWNED
        ${_trtmc_test_options}
      )
      if(NOT TARGET ${_trtmc_test_name})
        continue()
      endif()

      _trtmc_manifest_csv_to_list("${_trtmc_link_targets_csv}" _trtmc_link_targets)
      if(_trtmc_link_targets)
        target_link_libraries(${_trtmc_test_name} PRIVATE ${_trtmc_link_targets})
      endif()

      _trtmc_manifest_csv_to_list("${_trtmc_extra_sources_csv}" _trtmc_extra_sources)
      foreach(_trtmc_extra_source IN LISTS _trtmc_extra_sources)
        target_sources(
          ${_trtmc_test_name}
          PRIVATE "src/runtime/models/${_trtmc_model}/${_trtmc_extra_source}"
        )
      endforeach()
    endforeach()
  endforeach()
endfunction()

# --- Tests ---

# Bundle / format
trtmc_add_test(test_bundle_format)
trtmc_add_test(test_bundle_materialization)
trtmc_add_test(test_bundle_e2e          NO_SRC_INCLUDE)
trtmc_add_test(test_bundle_view)
trtmc_add_test(test_pipeline_registry)
trtmc_add_test(test_model_plugin_loader MODEL_OWNED)
add_dependencies(test_model_plugin_loader trtmc_model_plugins)
trtmc_add_test(test_backend_loader)
trtmc_add_test(test_trt_version)
trtmc_add_test(test_runtime_measurement)
trtmc_add_test(test_distributed_runtime)

trtmc_add_test(test_optimized_runtime_host)
add_dependencies(test_optimized_runtime_host
  trtmc_test_optimized_runtime
  trtmc_test_wrong_optimized_runtime
  trtmc_test_wrong_optimized_runtime_abi
  trtmc_test_optimized_embedding
)
target_compile_definitions(test_optimized_runtime_host PRIVATE
  TRTMC_TEST_OPTIMIZED_PROVIDER_DSO="$<TARGET_FILE:trtmc_test_optimized_runtime>"
  TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_DSO="$<TARGET_FILE:trtmc_test_wrong_optimized_runtime>"
  TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_ABI_DSO="$<TARGET_FILE:trtmc_test_wrong_optimized_runtime_abi>"
  TRTMC_TEST_OPTIMIZED_EMBEDDING_DSO="$<TARGET_FILE:trtmc_test_optimized_embedding>"
)
if(_trtmc_python)
  add_test(
    NAME test_optimized_runtime_bundle_contract
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
      "${_trtmc_python}"
      "${PROJECT_SOURCE_DIR}/tests/builder/optimized_runtime_cpp_contract.py"
      --loader "$<TARGET_FILE:test_optimized_runtime_host>"
      --provider "$<TARGET_FILE:trtmc_test_optimized_runtime>"
  )
  add_custom_target(test_optimized_runtime_bundle_contract
    DEPENDS test_optimized_runtime_host trtmc_test_optimized_runtime)
  add_dependencies(trtmc_cpu_cpp_tests test_optimized_runtime_bundle_contract)
  set_tests_properties(
    test_optimized_runtime_bundle_contract
    PROPERTIES LABELS "platform;cpu"
  )
endif()
# C ABI
trtmc_add_test(test_c_abi_entry                NO_SRC_INCLUDE)
trtmc_add_test(test_cabi_batch                 NO_SRC_INCLUDE)

# Config / parsing
trtmc_add_test(test_text_parsers)
trtmc_add_test(test_json_helpers)
trtmc_add_test(test_data_dir)
trtmc_add_test(test_cli_args)
target_sources(test_cli_args PRIVATE src/cli/args.cpp)
trtmc_add_test(test_jsonl_dataset_and_output
  EXTRA_INCLUDES ${_trtmc_nlohmann_json_include}
)
trtmc_add_test(test_speech_session_cli_helpers)
trtmc_add_test(test_config_schema_registry)
trtmc_add_test(test_config_cli_support)
trtmc_add_test(test_config_schemas_triattention)

# TRT runtime core
trtmc_add_test(test_trt_logger REQUIRES_TRT)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_logger PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_trt_runtime_lifetime REQUIRES_TRT REQUIRES_GPU)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_runtime_lifetime PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_trt_module REQUIRES_TRT REQUIRES_GPU)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_trt_module PRIVATE trtmc_backend_trt)
endif()

# CUDA / device
trtmc_add_test(test_cuda_buffer REQUIRES_TRT REQUIRES_GPU)
trtmc_add_test(test_cuda_stream REQUIRES_TRT REQUIRES_GPU)
trtmc_add_test(test_cuda_graph REQUIRES_TRT REQUIRES_GPU)
if(TARGET trtmc_backend_trt)
  target_link_libraries(test_cuda_graph PRIVATE trtmc_backend_trt)
endif()
trtmc_add_test(test_device_tensor REQUIRES_GPU)

# Pipelines (public API)
trtmc_add_test(test_pipeline_api                NO_SRC_INCLUDE)
trtmc_add_model_manifest_tests()

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

# Diffusion
trtmc_add_test(test_diffusion_math)

# Audio
trtmc_add_test(test_wav_reader)

# IoMap tensor name mapping
trtmc_add_test(test_io_map)

# Decoder config parsing
trtmc_add_test(test_decoder_config)

# TVM-FFI plugin
if(_trtmc_has_tvm_ffi)
  trtmc_add_test(test_tvm_ffi_shape_spec REQUIRES_TRT)

  add_library(trtmc_test_tvm_ffi_runtime_kernel SHARED EXCLUDE_FROM_ALL
    tests/cpp/test_tvm_ffi_runtime_bindings.cpp
  )
  target_compile_definitions(trtmc_test_tvm_ffi_runtime_kernel
    PRIVATE TRTMC_TVM_FFI_BINDINGS_TEST_DSO=1)
  target_include_directories(trtmc_test_tvm_ffi_runtime_kernel
    SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_test_tvm_ffi_runtime_kernel
    PRIVATE ${TRTMC_TVM_FFI_LIBRARY})

  trtmc_add_test(test_tvm_ffi_runtime_bindings)
  add_dependencies(test_tvm_ffi_runtime_bindings trtmc_test_tvm_ffi_runtime_kernel)
  target_compile_definitions(test_tvm_ffi_runtime_bindings PRIVATE
    TRTMC_TEST_TVM_FFI_BINDINGS_DSO="$<TARGET_FILE:trtmc_test_tvm_ffi_runtime_kernel>")
  target_include_directories(test_tvm_ffi_runtime_bindings
    SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_runtime_bindings PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
endif()

trtmc_add_test(test_tvm_ffi_plugin REQUIRES_TRT REQUIRES_GPU)
if(TARGET test_tvm_ffi_plugin AND _trtmc_has_tvm_ffi)
  target_include_directories(test_tvm_ffi_plugin SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(test_tvm_ffi_plugin PRIVATE ${TRTMC_TVM_FFI_LIBRARY})
endif()

# TVM-FFI plugin V2 (self-contained test with IPluginV2DynamicExt)
trtmc_add_test(test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE REQUIRES_TRT REQUIRES_GPU)

# TVM-FFI module loader (loads a CUDA FlashInfer .so when configured)
trtmc_add_test(test_tvm_ffi_module_loader REQUIRES_TRT REQUIRES_GPU)
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

if(TARGET trtmc_benchmark_worker)
  install(TARGETS trtmc_benchmark_worker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

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
