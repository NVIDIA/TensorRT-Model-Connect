# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_qwen3_06b_edge_qualification LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "The qualified capsule requires the GNU C and C++ toolchain")
endif()

set(TRTMC_EDGE_LLM_SOURCE_DIR "" CACHE PATH "Pinned EdgeLLM v0.9.0 source")
set(TRTMC_EDGE_LLM_BUILD_DIR "" CACHE PATH "Pinned EdgeLLM v0.9.0 build")
set(TRTMC_EDGE_LLM_PLUGIN_LIBRARY "" CACHE FILEPATH "Pinned EdgeLLM TensorRT plugin")
set(TRTMC_TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT 10.16 include directory")
set(TRTMC_TENSORRT_LIBRARY "" CACHE FILEPATH "TensorRT 10.16 nvinfer library")
set(TRTMC_TENSORRT_VERSION "" CACHE STRING "Exact TensorRT version")
set(TRTMC_CUDA_INCLUDE_DIR "" CACHE PATH "CUDA 12.9 include directory")
set(TRTMC_CUDART_LIBRARY "" CACHE FILEPATH "CUDA 12.9 runtime library")
set(TRTMC_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH "CUDA 12.9 driver library or toolkit stub")
set(TRTMC_CUDA_VERSION "" CACHE STRING "Exact CUDA toolkit version")
set(TRTMC_NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH "Directory containing nlohmann/json.hpp")
set(TRTMC_MC_INCLUDE_DIR "" CACHE PATH "Installed directory containing trtmc/pipeline.h")
set(TRTMC_MC_CORE_LIBRARY "" CACHE FILEPATH "Installed libtrtmc_core.so")

foreach(_directory
    TRTMC_EDGE_LLM_SOURCE_DIR
    TRTMC_EDGE_LLM_BUILD_DIR
    TRTMC_TENSORRT_INCLUDE_DIR
    TRTMC_CUDA_INCLUDE_DIR
    TRTMC_NLOHMANN_JSON_INCLUDE_DIR
    TRTMC_MC_INCLUDE_DIR)
  if(NOT IS_DIRECTORY "${${_directory}}")
    message(FATAL_ERROR "${_directory} must name an existing directory")
  endif()
endforeach()
foreach(_file
    TRTMC_EDGE_LLM_PLUGIN_LIBRARY
    TRTMC_TENSORRT_LIBRARY
    TRTMC_CUDART_LIBRARY
    TRTMC_CUDA_DRIVER_LIBRARY
    TRTMC_MC_CORE_LIBRARY)
  if(NOT EXISTS "${${_file}}" OR IS_DIRECTORY "${${_file}}")
    message(FATAL_ERROR "${_file} must name an existing file")
  endif()
endforeach()
file(REAL_PATH "${TRTMC_EDGE_LLM_BUILD_DIR}" _edge_build_real)
file(REAL_PATH "${TRTMC_EDGE_LLM_PLUGIN_LIBRARY}" _edge_plugin_real)
file(REAL_PATH "${TRTMC_EDGE_LLM_SOURCE_DIR}" _edge_source_real)
string(FIND "${_edge_plugin_real}" "${_edge_build_real}/" _edge_plugin_prefix)
if(NOT _edge_plugin_prefix EQUAL 0)
  message(FATAL_ERROR "TRTMC_EDGE_LLM_PLUGIN_LIBRARY must come from TRTMC_EDGE_LLM_BUILD_DIR")
endif()
set(_edge_cache "${TRTMC_EDGE_LLM_BUILD_DIR}/CMakeCache.txt")
if(NOT EXISTS "${_edge_cache}")
  message(FATAL_ERROR "TRTMC_EDGE_LLM_BUILD_DIR must contain CMakeCache.txt")
endif()
file(STRINGS "${_edge_cache}" _edge_home_lines
  REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
list(LENGTH _edge_home_lines _edge_home_line_count)
if(NOT _edge_home_line_count EQUAL 1)
  message(FATAL_ERROR "EdgeLLM build cache must identify exactly one source directory")
endif()
list(GET _edge_home_lines 0 _edge_home_line)
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" _edge_cached_source "${_edge_home_line}")
file(REAL_PATH "${_edge_cached_source}" _edge_cached_source_real)
if(NOT _edge_cached_source_real STREQUAL _edge_source_real)
  message(FATAL_ERROR "TRTMC_EDGE_LLM_BUILD_DIR was not configured from TRTMC_EDGE_LLM_SOURCE_DIR")
endif()
get_filename_component(_trt_library_name "${TRTMC_TENSORRT_LIBRARY}" NAME)
get_filename_component(_cudart_library_name "${TRTMC_CUDART_LIBRARY}" NAME)
get_filename_component(_mc_library_name "${TRTMC_MC_CORE_LIBRARY}" NAME)
get_filename_component(_edge_plugin_name "${TRTMC_EDGE_LLM_PLUGIN_LIBRARY}" NAME)
if(NOT _edge_plugin_name STREQUAL "libNvInfer_edgellm_plugin.so.1.0")
  message(FATAL_ERROR "TRTMC_EDGE_LLM_PLUGIN_LIBRARY must be the pinned EdgeLLM plugin product")
endif()
if(NOT _trt_library_name MATCHES "^libnvinfer\\.so(\\.10([.][0-9]+)*)?$")
  message(FATAL_ERROR "TRTMC_TENSORRT_LIBRARY must be libnvinfer.so.10")
endif()
if(NOT _cudart_library_name MATCHES "^libcudart\\.so(\\.12([.][0-9]+)*)?$")
  message(FATAL_ERROR "TRTMC_CUDART_LIBRARY must be libcudart.so.12")
endif()
if(NOT _mc_library_name MATCHES "^libtrtmc_core\\.so([.][0-9]+)*$")
  message(FATAL_ERROR "TRTMC_MC_CORE_LIBRARY must be libtrtmc_core.so")
endif()
foreach(_header
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/runtime/llmInferenceRuntime.h"
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h"
    "${TRTMC_TENSORRT_INCLUDE_DIR}/NvInfer.h"
    "${TRTMC_CUDA_INCLUDE_DIR}/cuda_runtime_api.h"
    "${TRTMC_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp"
    "${TRTMC_MC_INCLUDE_DIR}/trtmc/pipeline.h")
  if(NOT EXISTS "${_header}")
    message(FATAL_ERROR "Required qualification header is unavailable: ${_header}")
  endif()
endforeach()
if(NOT TRTMC_TENSORRT_VERSION STREQUAL "10.16.1.11")
  message(FATAL_ERROR "Qualification requires TensorRT 10.16.1.11")
endif()
if(NOT TRTMC_CUDA_VERSION STREQUAL "12.9")
  message(FATAL_ERROR "Qualification requires CUDA 12.9")
endif()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${TRTMC_EDGE_LLM_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE _edge_git_result
  OUTPUT_VARIABLE _edge_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT _edge_git_result EQUAL 0 OR
   NOT _edge_commit STREQUAL "1ac0f2b99642045125e1c5ac7b109434ba3b36c7")
  message(FATAL_ERROR "Qualification requires the pinned EdgeLLM v0.9.0 commit")
endif()
file(READ "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h" _edge_version_header)
if(NOT _edge_version_header MATCHES "kRUNTIME_VERSION[^\"]*\"0\\.9\\.0\"")
  message(FATAL_ERROR "Qualification requires EdgeLLM runtime version 0.9.0")
endif()

set(_edge_core "")
foreach(_candidate
    "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/libedgellmCore.a"
    "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/Release/libedgellmCore.a"
    "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/RelWithDebInfo/libedgellmCore.a")
  if(EXISTS "${_candidate}")
    set(_edge_core "${_candidate}")
    break()
  endif()
endforeach()
if(NOT _edge_core)
  message(FATAL_ERROR "TRTMC_EDGE_LLM_BUILD_DIR does not contain libedgellmCore.a")
endif()

enable_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^12\\.9(\\.[0-9]+)?$")
  message(FATAL_ERROR "Qualification requires exact CUDA 12.9 nvcc")
endif()

add_executable(trtmc_edgellm_direct_runner direct_runner.cpp)
target_compile_definitions(trtmc_edgellm_direct_runner PRIVATE
  TRTMC_EDGE_LLM_PLUGIN_PATH="${TRTMC_EDGE_LLM_PLUGIN_LIBRARY}")
target_include_directories(trtmc_edgellm_direct_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp"
  "${TRTMC_TENSORRT_INCLUDE_DIR}"
  "${TRTMC_CUDA_INCLUDE_DIR}"
  "${TRTMC_NLOHMANN_JSON_INCLUDE_DIR}")
target_link_libraries(trtmc_edgellm_direct_runner PRIVATE
  -Wl,--whole-archive
  "${_edge_core}"
  -Wl,--no-whole-archive
  "${TRTMC_TENSORRT_LIBRARY}"
  "${TRTMC_CUDART_LIBRARY}"
  "${TRTMC_CUDA_DRIVER_LIBRARY}"
  ${CMAKE_DL_LIBS})
set_target_properties(trtmc_edgellm_direct_runner PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_ARCHITECTURES 80
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(trtmc_edgellm_mc_runner mc_runner.cpp)
target_include_directories(trtmc_edgellm_mc_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TRTMC_MC_INCLUDE_DIR}"
  "${TRTMC_TENSORRT_INCLUDE_DIR}"
  "${TRTMC_CUDA_INCLUDE_DIR}"
  "${TRTMC_NLOHMANN_JSON_INCLUDE_DIR}")
target_link_libraries(trtmc_edgellm_mc_runner PRIVATE
  "${TRTMC_MC_CORE_LIBRARY}"
  "${TRTMC_TENSORRT_LIBRARY}"
  "${TRTMC_CUDART_LIBRARY}")

foreach(_target trtmc_edgellm_direct_runner trtmc_edgellm_mc_runner)
  target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endforeach()
