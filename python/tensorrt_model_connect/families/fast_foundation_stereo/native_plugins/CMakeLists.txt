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
if(DEFINED ENV{CONDA_PREFIX})
  file(GLOB _trt_python_roots "$ENV{CONDA_PREFIX}/lib/python*/site-packages/tensorrt")
  file(GLOB _trt_python_lib_dirs "$ENV{CONDA_PREFIX}/lib/python*/site-packages/tensorrt_libs")
  list(APPEND _trt_roots "$ENV{CONDA_PREFIX}" ${_trt_python_roots})
endif()

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

add_library(trtmc_fast_foundation_stereo_native_plugin SHARED
  gwc_plugin.cu
  gwc_plugin_creator.cpp
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
