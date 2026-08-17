# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

# L4 is SM 8.9. Set this before CUDA compiler detection, while retaining PTX for
# forward-compatible JIT on newer GPUs. Callers can override the cache value.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual" CACHE STRING
    "CUDA architectures for the Fast Foundation Stereo TensorRT plugin")
endif()

project(trtmc_fast_foundation_stereo_native_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

set(_trt_roots)
if(DEFINED ENV{TENSORRT_ROOT})
  list(APPEND _trt_roots "$ENV{TENSORRT_ROOT}")
endif()
set(_trt_python_prefixes)
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _trt_python_prefixes "$ENV{CONDA_PREFIX}")
endif()
if(DEFINED ENV{VIRTUAL_ENV})
  list(APPEND _trt_python_prefixes "$ENV{VIRTUAL_ENV}")
endif()
foreach(_trt_python_prefix IN LISTS _trt_python_prefixes)
  file(GLOB _trt_python_roots
    "${_trt_python_prefix}/lib/python*/site-packages/tensorrt")
  file(GLOB _trt_python_lib_dirs_for_prefix
    "${_trt_python_prefix}/lib/python*/site-packages/tensorrt_libs")
  list(APPEND _trt_roots "${_trt_python_prefix}" ${_trt_python_roots})
  list(APPEND _trt_python_lib_dirs ${_trt_python_lib_dirs_for_prefix})
endforeach()

find_path(FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR NvInferRuntime.h
  HINTS ${_trt_roots}
  PATH_SUFFIXES include
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(FAST_FOUNDATION_STEREO_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so libnvinfer.so.11 libnvinfer.so.10
  HINTS ${_trt_roots} ${_trt_python_lib_dirs}
  PATH_SUFFIXES lib lib64
  PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR OR NOT FAST_FOUNDATION_STEREO_TRT_LIBRARY)
  message(FATAL_ERROR
    "Fast Foundation Stereo native plugin requires TensorRT headers and libnvinfer")
endif()

if(NOT DEFINED FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION OR
   NOT FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION MATCHES
     "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "Fast Foundation Stereo native plugin requires "
    "FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION=major.minor.patch.build")
endif()

set(_trt_version_header "${FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${_trt_version_header}")
  message(FATAL_ERROR
    "Cannot validate TensorRT C++ headers: ${_trt_version_header} does not exist")
endif()

function(_ffs_trt_version_component output suffix)
  file(STRINGS "${_trt_version_header}" _enterprise_lines
    REGEX "^[ \t]*#[ \t]*define[ \t]+TRT_${suffix}_ENTERPRISE[ \t]+[0-9]+")
  if(_enterprise_lines)
    list(GET _enterprise_lines 0 _version_line)
  else()
    file(STRINGS "${_trt_version_header}" _nv_lines
      REGEX "^[ \t]*#[ \t]*define[ \t]+NV_TENSORRT_${suffix}[ \t]+[0-9]+")
    if(NOT _nv_lines)
      message(FATAL_ERROR
        "Cannot validate TensorRT C++ headers in ${_trt_version_header}: "
        "missing numeric TRT_${suffix}_ENTERPRISE or NV_TENSORRT_${suffix}")
    endif()
    list(GET _nv_lines 0 _version_line)
  endif()
  string(REGEX REPLACE ".*[ \t]([0-9]+).*$" "\\1" _component "${_version_line}")
  set(${output} "${_component}" PARENT_SCOPE)
endfunction()

foreach(_component MAJOR MINOR PATCH BUILD)
  _ffs_trt_version_component(_trt_${_component} ${_component})
endforeach()
set(_trt_detected_version
  "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}.${_trt_BUILD}")
if(NOT _trt_detected_version STREQUAL FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION)
  message(FATAL_ERROR
    "Fast Foundation Stereo TensorRT header/runtime mismatch:\n"
    "  active runtime: ${FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION}\n"
    "  pinned libnvinfer: ${FAST_FOUNDATION_STEREO_TRT_LIBRARY}\n"
    "  selected headers: ${FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR} "
    "(${_trt_detected_version})\n"
    "Set TRTMC_TRT_INCLUDE_DIR or TRT_INC_DIR to TensorRT "
    "${FAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION} headers (direct CMake: "
    "-DFAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR=...).")
endif()

add_library(trtmc_fast_foundation_stereo_native_plugin SHARED
  full_volume_leaky_plugin.cu
  full_volume_leaky_plugin_creator.cpp
  geometry_volume_convc1_plugin.cu
  geometry_volume_convc1_plugin_creator.cpp
  gwc_plugin.cu
  gwc_plugin_creator.cpp
  post8_sum_plugin.cu
  post8_sum_plugin_creator.cpp
  spatial_attention_reduce_plugin.cu
  spatial_attention_reduce_plugin_creator.cpp
)
target_include_directories(trtmc_fast_foundation_stereo_native_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR}
)
target_link_libraries(trtmc_fast_foundation_stereo_native_plugin PRIVATE
  ${FAST_FOUNDATION_STEREO_TRT_LIBRARY}
  CUDA::cudart
)
target_compile_options(trtmc_fast_foundation_stereo_native_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
)

set_property(TARGET trtmc_fast_foundation_stereo_native_plugin
  PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

get_filename_component(_trt_library_dir "${FAST_FOUNDATION_STEREO_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_fast_foundation_stereo_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_fast_foundation_stereo_native_plugin
  BUILD_RPATH "${_trt_library_dir};${CUDAToolkit_LIBRARY_DIR}"
)
