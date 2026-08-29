# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_minimax_h3_native_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(_required_variable
    MINIMAX_H3_TRT_EXPECTED_VERSION
    MINIMAX_H3_TORCH_EXPECTED_VERSION
    MINIMAX_H3_TORCH_EXPECTED_CUDA_VERSION
    MINIMAX_H3_TORCH_EXPECTED_CXX11_ABI
    MINIMAX_H3_TORCH_ROOT
    MINIMAX_H3_CUDA_TOOLKIT_EXPECTED_VERSION
    MINIMAX_H3_CXX_COMPILER_EXPECTED
    MINIMAX_H3_CXX_COMPILER_EXPECTED_VERSION
    MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY)
  if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "MiniMax-H3 native plugin requires ${_required_variable}")
  endif()
endforeach()

get_filename_component(_expected_cxx_compiler
  "${MINIMAX_H3_CXX_COMPILER_EXPECTED}" REALPATH)
get_filename_component(_selected_cxx_compiler "${CMAKE_CXX_COMPILER}" REALPATH)
if(NOT _selected_cxx_compiler STREQUAL _expected_cxx_compiler OR
   NOT CMAKE_CXX_COMPILER_VERSION STREQUAL MINIMAX_H3_CXX_COMPILER_EXPECTED_VERSION)
  message(FATAL_ERROR
    "MiniMax-H3 C++ compiler mismatch:\n"
    "  expected: ${_expected_cxx_compiler} "
    "(${MINIMAX_H3_CXX_COMPILER_EXPECTED_VERSION})\n"
    "  selected: ${_selected_cxx_compiler} (${CMAKE_CXX_COMPILER_VERSION})")
endif()

if(NOT MINIMAX_H3_TRT_EXPECTED_VERSION MATCHES
   "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "MINIMAX_H3_TRT_EXPECTED_VERSION must be major.minor.patch.build")
endif()
if(NOT MINIMAX_H3_TORCH_EXPECTED_CXX11_ABI MATCHES "^[01]$")
  message(FATAL_ERROR "MINIMAX_H3_TORCH_EXPECTED_CXX11_ABI must be 0 or 1")
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_VERSION STREQUAL MINIMAX_H3_CUDA_TOOLKIT_EXPECTED_VERSION)
  message(FATAL_ERROR
    "MiniMax-H3 CUDA toolkit mismatch:\n"
    "  expected: ${MINIMAX_H3_CUDA_TOOLKIT_EXPECTED_VERSION}\n"
    "  selected: ${CUDAToolkit_VERSION}")
endif()

get_filename_component(_expected_torch_root "${MINIMAX_H3_TORCH_ROOT}" REALPATH)
find_package(Torch CONFIG REQUIRED
  PATHS "${_expected_torch_root}/share/cmake/Torch"
  NO_DEFAULT_PATH)
get_filename_component(_selected_torch_root "${TORCH_INSTALL_PREFIX}" REALPATH)
if(NOT _selected_torch_root STREQUAL _expected_torch_root)
  message(FATAL_ERROR
    "MiniMax-H3 Torch installation mismatch:\n"
    "  active Python package: ${_expected_torch_root}\n"
    "  selected C++ package: ${_selected_torch_root}")
endif()

set(_torch_version_file "${_selected_torch_root}/version.py")
if(NOT EXISTS "${_torch_version_file}")
  message(FATAL_ERROR "Cannot validate Torch package: ${_torch_version_file} is missing")
endif()
file(STRINGS "${_torch_version_file}" _torch_version_lines
  REGEX "^__version__ = '[^']+'$")
file(STRINGS "${_torch_version_file}" _torch_cuda_lines
  REGEX "^cuda: Optional\\[str\\] = '[^']+'$")
if(NOT _torch_version_lines OR NOT _torch_cuda_lines)
  message(FATAL_ERROR "Cannot read exact Torch/CUDA identity from ${_torch_version_file}")
endif()
list(GET _torch_version_lines 0 _torch_version_line)
list(GET _torch_cuda_lines 0 _torch_cuda_line)
string(REGEX REPLACE "^__version__ = '([^']+)'$" "\\1"
  _torch_detected_version "${_torch_version_line}")
string(REGEX REPLACE "^cuda: Optional\\[str\\] = '([^']+)'$" "\\1"
  _torch_detected_cuda_version "${_torch_cuda_line}")
if(NOT _torch_detected_version STREQUAL MINIMAX_H3_TORCH_EXPECTED_VERSION)
  message(FATAL_ERROR
    "MiniMax-H3 Torch version mismatch: expected "
    "${MINIMAX_H3_TORCH_EXPECTED_VERSION}, selected ${_torch_detected_version}")
endif()
if(NOT _torch_detected_cuda_version STREQUAL MINIMAX_H3_TORCH_EXPECTED_CUDA_VERSION)
  message(FATAL_ERROR
    "MiniMax-H3 Torch CUDA version mismatch: expected "
    "${MINIMAX_H3_TORCH_EXPECTED_CUDA_VERSION}, selected ${_torch_detected_cuda_version}")
endif()

set(_trt_roots)
if(DEFINED ENV{TENSORRT_ROOT})
  list(APPEND _trt_roots "$ENV{TENSORRT_ROOT}")
endif()
foreach(_trt_python_prefix IN ITEMS "$ENV{CONDA_PREFIX}" "$ENV{VIRTUAL_ENV}")
  if(NOT "${_trt_python_prefix}" STREQUAL "")
    file(GLOB _trt_python_roots
      "${_trt_python_prefix}/lib/python*/site-packages/tensorrt")
    file(GLOB _trt_python_lib_dirs_for_prefix
      "${_trt_python_prefix}/lib/python*/site-packages/tensorrt_libs")
    list(APPEND _trt_roots "${_trt_python_prefix}" ${_trt_python_roots})
    list(APPEND _trt_python_lib_dirs ${_trt_python_lib_dirs_for_prefix})
  endif()
endforeach()

find_path(MINIMAX_H3_TRT_INCLUDE_DIR NvInferRuntime.h
  HINTS ${_trt_roots}
  PATH_SUFFIXES include
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include)
find_library(MINIMAX_H3_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so libnvinfer.so.11 libnvinfer.so.10
  HINTS ${_trt_roots} ${_trt_python_lib_dirs}
  PATH_SUFFIXES lib lib64
  PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib)
if(NOT MINIMAX_H3_TRT_INCLUDE_DIR OR NOT MINIMAX_H3_TRT_LIBRARY)
  message(FATAL_ERROR
    "MiniMax-H3 native plugin requires TensorRT headers and libnvinfer")
endif()

set(_trt_version_header "${MINIMAX_H3_TRT_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${_trt_version_header}")
  message(FATAL_ERROR
    "Cannot validate TensorRT C++ headers: ${_trt_version_header} does not exist")
endif()

function(_minimax_h3_trt_version_component output suffix)
  file(STRINGS "${_trt_version_header}" _enterprise_lines
    REGEX "^[ \\t]*#[ \\t]*define[ \\t]+TRT_${suffix}_ENTERPRISE[ \\t]+[0-9]+")
  if(_enterprise_lines)
    list(GET _enterprise_lines 0 _version_line)
  else()
    file(STRINGS "${_trt_version_header}" _nv_lines
      REGEX "^[ \\t]*#[ \\t]*define[ \\t]+NV_TENSORRT_${suffix}[ \\t]+[0-9]+")
    if(NOT _nv_lines)
      message(FATAL_ERROR
        "Cannot validate TensorRT C++ headers in ${_trt_version_header}: "
        "missing numeric TRT_${suffix}_ENTERPRISE or NV_TENSORRT_${suffix}")
    endif()
    list(GET _nv_lines 0 _version_line)
  endif()
  string(REGEX REPLACE ".*[ \\t]([0-9]+).*$" "\\1" _component "${_version_line}")
  set(${output} "${_component}" PARENT_SCOPE)
endfunction()

foreach(_component MAJOR MINOR PATCH BUILD)
  _minimax_h3_trt_version_component(_trt_${_component} ${_component})
endforeach()
set(_trt_detected_version
  "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}.${_trt_BUILD}")
if(NOT _trt_detected_version STREQUAL MINIMAX_H3_TRT_EXPECTED_VERSION)
  message(FATAL_ERROR
    "MiniMax-H3 TensorRT header/runtime mismatch:\n"
    "  active runtime: ${MINIMAX_H3_TRT_EXPECTED_VERSION}\n"
    "  pinned libnvinfer: ${MINIMAX_H3_TRT_LIBRARY}\n"
    "  selected headers: ${MINIMAX_H3_TRT_INCLUDE_DIR} (${_trt_detected_version})")
endif()

add_library(trtmc_minimax_h3_native_plugin SHARED
  layer_norm_plugin.cpp
  linear_plugin.cpp
  patch_embed_plugin.cpp
  plugin_creator.cpp
  vision_attention_plugin.cpp)
target_include_directories(trtmc_minimax_h3_native_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MINIMAX_H3_TRT_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_minimax_h3_native_plugin PRIVATE
  ${MINIMAX_H3_TRT_LIBRARY}
  CUDA::cudart
  ${TORCH_LIBRARIES})
target_compile_definitions(trtmc_minimax_h3_native_plugin PRIVATE
  "_GLIBCXX_USE_CXX11_ABI=${MINIMAX_H3_TORCH_EXPECTED_CXX11_ABI}"
  "TRTMC_MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY=\"${MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY}\"")
target_compile_options(trtmc_minimax_h3_native_plugin PRIVATE
  -O3 -Wall -Wextra -Wpedantic)

get_filename_component(_trt_library_dir "${MINIMAX_H3_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_minimax_h3_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_minimax_h3_native_plugin
  BUILD_RPATH
    "${_trt_library_dir};${_selected_torch_root}/lib;${CUDAToolkit_LIBRARY_DIR}")
