# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_qwen3_06b_fp16_edge_llm_v09_trt11_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "This qualified capsule requires the GNU C++ toolchain; got "
    "${CMAKE_CXX_COMPILER_ID}")
endif()

option(TRTMC_QWEN_EDGE_FAKE_RUNTIME
  "Build the capsule ABI/metadata contract without CUDA or Edge-LLM" OFF)
set(TRTMC_SDK_INCLUDE_DIRS "" CACHE STRING
  "Semicolon-separated Model Connect source include roots or installed private SDK root")
set(TRTMC_NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH
  "Directory containing nlohmann/json.hpp")
set(TRTMC_QWEN_EDGE_MANIFEST_SHA256 "" CACHE STRING
  "Normalized implementation manifest digest compiled into this DSO")

set(_factory_header_found FALSE)
set(_pipeline_header "")
foreach(_sdk_root IN LISTS TRTMC_SDK_INCLUDE_DIRS)
  if(EXISTS "${_sdk_root}/runtime/providers/optimized_runtime_factory.h")
    set(_factory_header_found TRUE)
  endif()
  if(NOT _pipeline_header AND EXISTS "${_sdk_root}/trtmc/pipeline.h")
    set(_pipeline_header "${_sdk_root}/trtmc/pipeline.h")
  endif()
endforeach()
if(NOT _factory_header_found OR NOT _pipeline_header)
  message(FATAL_ERROR
    "TRTMC_SDK_INCLUDE_DIRS must resolve runtime/providers/optimized_runtime_factory.h "
    "and trtmc/pipeline.h")
endif()
if(NOT TRTMC_NLOHMANN_JSON_INCLUDE_DIR OR
   NOT EXISTS "${TRTMC_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "TRTMC_NLOHMANN_JSON_INCLUDE_DIR must contain nlohmann/json.hpp")
endif()
string(LENGTH "${TRTMC_QWEN_EDGE_MANIFEST_SHA256}" _manifest_digest_length)
if(NOT _manifest_digest_length EQUAL 64 OR
   NOT TRTMC_QWEN_EDGE_MANIFEST_SHA256 MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR
    "TRTMC_QWEN_EDGE_MANIFEST_SHA256 must be a lowercase SHA-256 digest")
endif()

set(_target trtmc_impl_qwen3_1_7b_fp16_tensorrt_edge_llm_v0_9_trt11)
add_library(${_target} SHARED adapter.cpp)
target_compile_definitions(${_target} PRIVATE
  TRTMC_QWEN_EDGE_MANIFEST_SHA256="${TRTMC_QWEN_EDGE_MANIFEST_SHA256}")
target_include_directories(${_target} PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  ${TRTMC_SDK_INCLUDE_DIRS}
  "${TRTMC_NLOHMANN_JSON_INCLUDE_DIR}")
target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(${_target} PRIVATE ${CMAKE_DL_LIBS})
target_link_options(${_target} PRIVATE
  -Wl,--exclude-libs,ALL
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
set_target_properties(${_target} PROPERTIES
  OUTPUT_NAME trtmc_impl_qwen3_1_7b_fp16_tensorrt_edge_llm_v0_9_trt11
  SKIP_BUILD_RPATH TRUE
  INSTALL_RPATH ""
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

if(TRTMC_QWEN_EDGE_FAKE_RUNTIME)
  set(TRTMC_QWEN_EDGE_FAKE_TENSORRT_MAJOR 11 CACHE STRING
    "Fake loaded TensorRT major version for capsule contract tests")
  set(TRTMC_QWEN_EDGE_FAKE_TENSORRT_MINOR 2 CACHE STRING
    "Fake loaded TensorRT minor version for capsule contract tests")
  set(TRTMC_QWEN_EDGE_FAKE_TENSORRT_PATCH 0 CACHE STRING
    "Fake loaded TensorRT patch version for capsule contract tests")
  set(TRTMC_QWEN_EDGE_FAKE_TENSORRT_BUILD 113 CACHE STRING
    "Fake loaded TensorRT build version for capsule contract tests")
  set(TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION 13030 CACHE STRING
    "Fake loaded CUDA runtime integer version for capsule contract tests")
  foreach(_component MAJOR MINOR PATCH BUILD)
    if(NOT TRTMC_QWEN_EDGE_FAKE_TENSORRT_${_component} MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "TRTMC_QWEN_EDGE_FAKE_TENSORRT_${_component} must be a non-negative integer")
    endif()
  endforeach()
  if(NOT TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION must be a non-negative integer")
  endif()
  target_compile_definitions(${_target} PRIVATE
    TRTMC_QWEN_EDGE_FAKE_RUNTIME=1
    TRTMC_QWEN_EDGE_FAKE_TENSORRT_MAJOR=${TRTMC_QWEN_EDGE_FAKE_TENSORRT_MAJOR}
    TRTMC_QWEN_EDGE_FAKE_TENSORRT_MINOR=${TRTMC_QWEN_EDGE_FAKE_TENSORRT_MINOR}
    TRTMC_QWEN_EDGE_FAKE_TENSORRT_PATCH=${TRTMC_QWEN_EDGE_FAKE_TENSORRT_PATCH}
    TRTMC_QWEN_EDGE_FAKE_TENSORRT_BUILD=${TRTMC_QWEN_EDGE_FAKE_TENSORRT_BUILD}
    TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION=${TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION})
else()
  set(TRTMC_EDGE_LLM_SOURCE_DIR "" CACHE PATH
    "Pinned TensorRT Edge-LLM v0.9.0 source directory")
  set(TRTMC_EDGE_LLM_BUILD_DIR "" CACHE PATH
    "Complete TensorRT Edge-LLM v0.9.0 build directory")
  set(TRTMC_TENSORRT_INCLUDE_DIR "" CACHE PATH
    "Directory containing NvInfer.h and NvInferVersion.h")
  set(TRTMC_TENSORRT_LIBRARY "" CACHE FILEPATH "Exact TensorRT nvinfer library")
  set(TRTMC_TENSORRT_VERSION "" CACHE STRING "Exact TensorRT major.minor.patch.build")
  set(TRTMC_CUDA_INCLUDE_DIR "" CACHE PATH "Directory containing cuda_runtime_api.h")
  set(TRTMC_CUDART_LIBRARY "" CACHE FILEPATH "CUDA runtime library")
  set(TRTMC_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH
    "CUDA driver library or toolkit stub from the selected CUDA installation")
  set(TRTMC_CUDA_VERSION "" CACHE STRING "Exact CUDA toolkit major.minor version")

  foreach(_required
      TRTMC_EDGE_LLM_SOURCE_DIR
      TRTMC_EDGE_LLM_BUILD_DIR
      TRTMC_TENSORRT_INCLUDE_DIR
      TRTMC_TENSORRT_LIBRARY
      TRTMC_TENSORRT_VERSION
      TRTMC_CUDA_INCLUDE_DIR
      TRTMC_CUDART_LIBRARY
      TRTMC_CUDA_DRIVER_LIBRARY
      TRTMC_CUDA_VERSION)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
      message(FATAL_ERROR "${_required} is required for the real capsule runtime")
    endif()
  endforeach()
  if(NOT TRTMC_TENSORRT_VERSION STREQUAL "11.2.0.113")
    message(FATAL_ERROR
      "This capsule supports TensorRT 11.2.0.113, got ${TRTMC_TENSORRT_VERSION}")
  endif()
  if(NOT TRTMC_CUDA_VERSION STREQUAL "13.3")
    message(FATAL_ERROR
      "This capsule supports CUDA 13.3, got ${TRTMC_CUDA_VERSION}")
  endif()
  foreach(_header
      "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/runtime/llmInferenceRuntime.h"
      "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h"
      "${TRTMC_TENSORRT_INCLUDE_DIR}/NvInfer.h"
      "${TRTMC_TENSORRT_INCLUDE_DIR}/NvInferVersion.h"
      "${TRTMC_CUDA_INCLUDE_DIR}/cuda.h"
      "${TRTMC_CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
    if(NOT EXISTS "${_header}")
      message(FATAL_ERROR "Required capsule build header is unavailable: ${_header}")
    endif()
  endforeach()
  foreach(_library
      TRTMC_TENSORRT_LIBRARY
      TRTMC_CUDART_LIBRARY
      TRTMC_CUDA_DRIVER_LIBRARY)
    if(NOT EXISTS "${${_library}}")
      message(FATAL_ERROR "Required capsule build library is unavailable: ${${_library}}")
    endif()
  endforeach()

  function(_qwen_edge_header_define OUT_VAR DEFINE_NAME)
    file(READ "${TRTMC_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _version_header)
    string(REGEX MATCH
      "#define[ \t]+${DEFINE_NAME}[ \t]+([A-Za-z0-9_]+)"
      _define_match "${_version_header}")
    if(NOT _define_match)
      set(${OUT_VAR} "" PARENT_SCOPE)
      return()
    endif()
    set(_value "${CMAKE_MATCH_1}")
    if(NOT _value MATCHES "^[0-9]+$")
      string(REGEX MATCH "#define[ \t]+${_value}[ \t]+([0-9]+)"
        _alias_match "${_version_header}")
      if(_alias_match)
        set(_value "${CMAKE_MATCH_1}")
      endif()
    endif()
    if(_value MATCHES "^[0-9]+$")
      set(${OUT_VAR} "${_value}" PARENT_SCOPE)
    else()
      set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
  endfunction()
  _qwen_edge_header_define(_trt_major TRT_MAJOR_ENTERPRISE)
  _qwen_edge_header_define(_trt_minor TRT_MINOR_ENTERPRISE)
  _qwen_edge_header_define(_trt_patch TRT_PATCH_ENTERPRISE)
  _qwen_edge_header_define(_trt_build TRT_BUILD_ENTERPRISE)
  if("${_trt_major}" STREQUAL "" OR "${_trt_minor}" STREQUAL "" OR
     "${_trt_patch}" STREQUAL "" OR "${_trt_build}" STREQUAL "")
    _qwen_edge_header_define(_trt_major NV_TENSORRT_MAJOR)
    _qwen_edge_header_define(_trt_minor NV_TENSORRT_MINOR)
    _qwen_edge_header_define(_trt_patch NV_TENSORRT_PATCH)
    _qwen_edge_header_define(_trt_build NV_TENSORRT_BUILD)
  endif()
  set(_header_trt_version
    "${_trt_major}.${_trt_minor}.${_trt_patch}.${_trt_build}")
  if(NOT _header_trt_version STREQUAL TRTMC_TENSORRT_VERSION)
    message(FATAL_ERROR
      "TensorRT header version ${_header_trt_version} does not match "
      "TRTMC_TENSORRT_VERSION=${TRTMC_TENSORRT_VERSION}")
  endif()
  file(READ "${TRTMC_CUDA_INCLUDE_DIR}/cuda.h" _cuda_version_header)
  if(NOT _cuda_version_header MATCHES "#define[ \t]+CUDA_VERSION[ \t]+13030([^0-9]|$)")
    message(FATAL_ERROR "This capsule requires exact CUDA 13.3 headers")
  endif()
  file(REAL_PATH "${TRTMC_CUDA_INCLUDE_DIR}" _cuda_include_real)
  get_filename_component(_cuda_target_root "${_cuda_include_real}" DIRECTORY)
  foreach(_cuda_library TRTMC_CUDART_LIBRARY TRTMC_CUDA_DRIVER_LIBRARY)
    file(REAL_PATH "${${_cuda_library}}" _cuda_library_real)
    string(FIND "${_cuda_library_real}" "${_cuda_target_root}/" _cuda_root_prefix)
    if(NOT _cuda_root_prefix EQUAL 0)
      message(FATAL_ERROR
        "${_cuda_library} must come from the same CUDA 13.3 installation as "
        "TRTMC_CUDA_INCLUDE_DIR")
    endif()
  endforeach()

  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${TRTMC_EDGE_LLM_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE _edge_git_result
    OUTPUT_VARIABLE _edge_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT _edge_git_result EQUAL 0 OR
     NOT _edge_commit STREQUAL "1ac0f2b99642045125e1c5ac7b109434ba3b36c7")
    message(FATAL_ERROR
      "The capsule requires TensorRT Edge-LLM commit "
      "1ac0f2b99642045125e1c5ac7b109434ba3b36c7; got ${_edge_commit}")
  endif()
  file(READ "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h" _edge_version_header)
  if(NOT _edge_version_header MATCHES "kRUNTIME_VERSION[^\"]*\"0\\.9\\.0\"")
    message(FATAL_ERROR "The capsule requires TensorRT Edge-LLM runtime version 0.9.0")
  endif()

  set(_edge_resolved FALSE)
  foreach(_config "" Release RelWithDebInfo Debug)
    if(_config)
      set(_suffix "/${_config}")
    else()
      set(_suffix "")
    endif()
    set(_edge_core
      "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp${_suffix}/libedgellmCore.a")
    set(_edge_plugin
      "${TRTMC_EDGE_LLM_BUILD_DIR}${_suffix}/libNvInfer_edgellm_plugin.so.1.0")
    if(EXISTS "${_edge_core}" AND EXISTS "${_edge_plugin}")
      set(_edge_resolved TRUE)
      break()
    endif()
  endforeach()
  if(NOT _edge_resolved)
    message(FATAL_ERROR
      "TRTMC_EDGE_LLM_BUILD_DIR is not a complete pinned build; expected "
      "libedgellmCore.a and libNvInfer_edgellm_plugin.so.1.0")
  endif()

  enable_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^13\\.3(\\.[0-9]+)?$")
    message(FATAL_ERROR
      "This capsule requires exact CUDA 13.3 nvcc, got "
      "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  target_include_directories(${_target} PRIVATE
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp"
    "${TRTMC_TENSORRT_INCLUDE_DIR}"
    "${TRTMC_CUDA_INCLUDE_DIR}")
  target_link_libraries(${_target} PRIVATE
    -Wl,--whole-archive
    "${_edge_core}"
    -Wl,--no-whole-archive
    "${TRTMC_TENSORRT_LIBRARY}"
    "${TRTMC_CUDART_LIBRARY}"
    "${TRTMC_CUDA_DRIVER_LIBRARY}")
  set_target_properties(${_target} PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET ${_target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${_edge_plugin}"
      "$<TARGET_FILE_DIR:${_target}>/libNvInfer_edgellm_plugin.so"
    COMMENT "Staging the pinned Edge-LLM TensorRT plugin beside the capsule DSO")
endif()
