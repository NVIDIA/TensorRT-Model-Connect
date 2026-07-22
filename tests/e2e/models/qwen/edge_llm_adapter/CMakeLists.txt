# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(trtmc_qwen_edgellm_performance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "Qwen EdgeLLM performance runners require the GNU C++ toolchain")
endif()

set(TRTMC_EDGE_LLM_SOURCE_DIR "" CACHE PATH "Pinned TensorRT Edge-LLM source")
set(TRTMC_EDGE_LLM_BUILD_DIR "" CACHE PATH "Pinned TensorRT Edge-LLM build")
set(TRTMC_EDGE_LLM_SOURCE "" CACHE STRING "Pinned TensorRT Edge-LLM source URL")
set(TRTMC_EDGE_LLM_VERSION "" CACHE STRING "Pinned TensorRT Edge-LLM version")
set(TRTMC_EDGE_LLM_COMMIT "" CACHE STRING "Pinned TensorRT Edge-LLM commit")
set(TRTMC_TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT include directory")
set(TRTMC_TENSORRT_LIBRARY "" CACHE FILEPATH "TensorRT nvinfer library")
set(TRTMC_TENSORRT_VERSION "" CACHE STRING "Pinned TensorRT version")
set(TRTMC_TENSORRT_MAJOR "" CACHE STRING "Pinned TensorRT major version")
set(TRTMC_TENSORRT_MINOR "" CACHE STRING "Pinned TensorRT minor version")
set(TRTMC_TENSORRT_PATCH "" CACHE STRING "Pinned TensorRT patch version")
set(TRTMC_TENSORRT_BUILD "" CACHE STRING "Pinned TensorRT build version")
set(TRTMC_CUDA_INCLUDE_DIR "" CACHE PATH "CUDA include directory")
set(TRTMC_CUDART_LIBRARY "" CACHE FILEPATH "CUDA runtime library")
set(TRTMC_CUDA_DRIVER_LIBRARY "" CACHE FILEPATH "CUDA driver library or toolkit stub")
set(TRTMC_CUDA_VERSION "" CACHE STRING "Pinned CUDA major.minor version")
set(TRTMC_CUDA_RUNTIME_VERSION "" CACHE STRING "Pinned CUDART_VERSION value")
set(TRTMC_MC_INCLUDE_DIR "" CACHE PATH "Installed directory containing trtmc/pipeline.h")
set(TRTMC_MC_CORE_LIBRARY "" CACHE FILEPATH "Installed libtrtmc_core.so")

foreach(_pin
    TRTMC_EDGE_LLM_SOURCE
    TRTMC_EDGE_LLM_VERSION
    TRTMC_EDGE_LLM_COMMIT
    TRTMC_TENSORRT_VERSION
    TRTMC_TENSORRT_MAJOR
    TRTMC_TENSORRT_MINOR
    TRTMC_TENSORRT_PATCH
    TRTMC_TENSORRT_BUILD
    TRTMC_CUDA_VERSION
    TRTMC_CUDA_RUNTIME_VERSION)
  if("${${_pin}}" STREQUAL "")
    message(FATAL_ERROR "${_pin} must be supplied from the model-owned dependency.lock")
  endif()
endforeach()
string(LENGTH "${TRTMC_EDGE_LLM_COMMIT}" _edge_commit_length)
if(NOT _edge_commit_length EQUAL 40 OR NOT TRTMC_EDGE_LLM_COMMIT MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "TRTMC_EDGE_LLM_COMMIT must be a lowercase Git commit")
endif()
if(NOT TRTMC_TENSORRT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "TRTMC_TENSORRT_VERSION must have major.minor.patch.build")
endif()
foreach(_part
    TRTMC_TENSORRT_MAJOR
    TRTMC_TENSORRT_MINOR
    TRTMC_TENSORRT_PATCH
    TRTMC_TENSORRT_BUILD
    TRTMC_CUDA_RUNTIME_VERSION)
  if(NOT ${_part} MATCHES "^[0-9]+$")
    message(FATAL_ERROR "${_part} must be numeric")
  endif()
endforeach()
string(REPLACE "." "[.]" _cuda_version_pattern "${TRTMC_CUDA_VERSION}")
if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^${_cuda_version_pattern}(\\.[0-9]+)?$")
  message(FATAL_ERROR
    "Qwen EdgeLLM performance runners require CUDA ${TRTMC_CUDA_VERSION} nvcc")
endif()

foreach(_directory
    TRTMC_EDGE_LLM_SOURCE_DIR
    TRTMC_EDGE_LLM_BUILD_DIR
    TRTMC_TENSORRT_INCLUDE_DIR
    TRTMC_CUDA_INCLUDE_DIR
    TRTMC_MC_INCLUDE_DIR)
  if(NOT IS_DIRECTORY "${${_directory}}")
    message(FATAL_ERROR "${_directory} must name an existing directory")
  endif()
endforeach()
foreach(_file
    TRTMC_TENSORRT_LIBRARY
    TRTMC_CUDART_LIBRARY
    TRTMC_CUDA_DRIVER_LIBRARY
    TRTMC_MC_CORE_LIBRARY)
  if(NOT EXISTS "${${_file}}" OR IS_DIRECTORY "${${_file}}")
    message(FATAL_ERROR "${_file} must name an existing file")
  endif()
endforeach()
foreach(_header
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/runtime/llmInferenceRuntime.h"
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h"
    "${TRTMC_EDGE_LLM_SOURCE_DIR}/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
    "${TRTMC_TENSORRT_INCLUDE_DIR}/NvInfer.h"
    "${TRTMC_CUDA_INCLUDE_DIR}/cuda_runtime_api.h"
    "${TRTMC_MC_INCLUDE_DIR}/trtmc/pipeline.h")
  if(NOT EXISTS "${_header}")
    message(FATAL_ERROR "Required performance-runner header is unavailable: ${_header}")
  endif()
endforeach()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${TRTMC_EDGE_LLM_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE _edge_git_result
  OUTPUT_VARIABLE _edge_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT _edge_git_result EQUAL 0 OR NOT _edge_commit STREQUAL TRTMC_EDGE_LLM_COMMIT)
  message(FATAL_ERROR
    "Performance qualification requires EdgeLLM ${TRTMC_EDGE_LLM_COMMIT}")
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${TRTMC_EDGE_LLM_SOURCE_DIR}" remote get-url origin
  RESULT_VARIABLE _edge_origin_result
  OUTPUT_VARIABLE _edge_origin
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT _edge_origin_result EQUAL 0 OR NOT _edge_origin STREQUAL TRTMC_EDGE_LLM_SOURCE)
  message(FATAL_ERROR
    "Performance qualification requires EdgeLLM source ${TRTMC_EDGE_LLM_SOURCE}")
endif()
file(READ "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/common/version.h" _edge_version_header)
string(REPLACE "." "[.]" _edge_version_pattern "${TRTMC_EDGE_LLM_VERSION}")
if(NOT _edge_version_header MATCHES
   "kRUNTIME_VERSION[^\"]*\"${_edge_version_pattern}\"")
  message(FATAL_ERROR
    "Performance qualification requires EdgeLLM runtime version ${TRTMC_EDGE_LLM_VERSION}")
endif()

set(_edge_core "")
foreach(_configuration "" Release RelWithDebInfo Debug)
  if(_configuration)
    set(_candidate "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/${_configuration}/libedgellmCore.a")
  else()
    set(_candidate "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/libedgellmCore.a")
  endif()
  if(EXISTS "${_candidate}")
    set(_edge_core "${_candidate}")
    break()
  endif()
endforeach()
if(NOT _edge_core)
  message(FATAL_ERROR "TRTMC_EDGE_LLM_BUILD_DIR does not contain libedgellmCore.a")
endif()

add_executable(trtmc_qwen_edgellm_direct_performance
  direct_performance_runner.cpp
  performance_device_link_stub.cu)
target_include_directories(trtmc_qwen_edgellm_direct_performance PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp"
  "${TRTMC_EDGE_LLM_SOURCE_DIR}/3rdParty/nlohmannJson/include"
  "${TRTMC_TENSORRT_INCLUDE_DIR}"
  "${TRTMC_CUDA_INCLUDE_DIR}")
target_link_libraries(trtmc_qwen_edgellm_direct_performance PRIVATE
  -Wl,--whole-archive
  "${_edge_core}"
  -Wl,--no-whole-archive
  "${TRTMC_TENSORRT_LIBRARY}"
  "${TRTMC_CUDART_LIBRARY}"
  "${TRTMC_CUDA_DRIVER_LIBRARY}"
  ${CMAKE_DL_LIBS})
set_target_properties(trtmc_qwen_edgellm_direct_performance PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_ARCHITECTURES 80
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(trtmc_qwen_edgellm_mc_performance mc_performance_runner.cpp)
target_include_directories(trtmc_qwen_edgellm_mc_performance PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TRTMC_EDGE_LLM_SOURCE_DIR}/3rdParty/nlohmannJson/include"
  "${TRTMC_MC_INCLUDE_DIR}"
  "${TRTMC_TENSORRT_INCLUDE_DIR}"
  "${TRTMC_CUDA_INCLUDE_DIR}")
target_link_libraries(trtmc_qwen_edgellm_mc_performance PRIVATE
  "${TRTMC_MC_CORE_LIBRARY}"
  "${TRTMC_TENSORRT_LIBRARY}"
  "${TRTMC_CUDART_LIBRARY}")

get_filename_component(_mc_library_dir "${TRTMC_MC_CORE_LIBRARY}" DIRECTORY)
get_filename_component(_trt_library_dir "${TRTMC_TENSORRT_LIBRARY}" DIRECTORY)
get_filename_component(_cuda_library_dir "${TRTMC_CUDART_LIBRARY}" DIRECTORY)
foreach(_target
    trtmc_qwen_edgellm_direct_performance
    trtmc_qwen_edgellm_mc_performance)
  target_compile_definitions(${_target} PRIVATE
    TRTMC_EXPECTED_TENSORRT_MAJOR=${TRTMC_TENSORRT_MAJOR}
    TRTMC_EXPECTED_TENSORRT_MINOR=${TRTMC_TENSORRT_MINOR}
    TRTMC_EXPECTED_TENSORRT_PATCH=${TRTMC_TENSORRT_PATCH}
    TRTMC_EXPECTED_TENSORRT_BUILD=${TRTMC_TENSORRT_BUILD}
    TRTMC_EXPECTED_TENSORRT_VERSION="${TRTMC_TENSORRT_VERSION}"
    TRTMC_EXPECTED_CUDA_RUNTIME_VERSION=${TRTMC_CUDA_RUNTIME_VERSION})
  target_compile_options(${_target} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
  target_link_options(${_target} PRIVATE -Wl,--no-undefined)
  set_target_properties(${_target} PROPERTIES
    BUILD_RPATH "${_mc_library_dir};${_trt_library_dir};${_cuda_library_dir}")
endforeach()
