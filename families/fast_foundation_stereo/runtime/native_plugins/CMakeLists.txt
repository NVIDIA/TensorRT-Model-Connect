# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_fast_foundation_stereo_native_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_path(FAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR
  NAMES NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
  REQUIRED
)
find_library(FAST_FOUNDATION_STEREO_TRT_LIBRARY
  NAMES libnvinfer.so.11
  PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
  REQUIRED
)

add_library(trtmc_fast_foundation_stereo_native_plugin SHARED
  full_volume_leaky_plugin.cu
  geometry_volume_convc1_plugin.cu
  gwc_plugin.cu
  plugin_creators.cpp
  post8_sum_plugin.cu
  spatial_attention_reduce_plugin.cu
)
set_source_files_properties(plugin_creators.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-deprecated-declarations"
)
set_source_files_properties(
  full_volume_leaky_plugin.cu
  geometry_volume_convc1_plugin.cu
  gwc_plugin.cu
  post8_sum_plugin.cu
  spatial_attention_reduce_plugin.cu
  PROPERTIES COMPILE_OPTIONS "-diag-suppress=997"
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
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Wall;-Wextra>"
)

set_property(TARGET trtmc_fast_foundation_stereo_native_plugin
  PROPERTY CUDA_ARCHITECTURES "89-real;89-virtual")

get_filename_component(_trt_library_dir "${FAST_FOUNDATION_STEREO_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_fast_foundation_stereo_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_fast_foundation_stereo_native_plugin
  BUILD_RPATH "${_trt_library_dir};${CUDAToolkit_LIBRARY_DIR}"
)
