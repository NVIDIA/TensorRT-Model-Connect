# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_sam2_hoi_native_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

find_path(SAM2_HOI_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(SAM2_HOI_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT SAM2_HOI_TRT_INCLUDE_DIR OR NOT SAM2_HOI_TRT_LIBRARY)
  message(FATAL_ERROR "SAM2 HOI native plugin requires TensorRT headers and libnvinfer")
endif()

add_library(trtmc_sam2_hoi_native_plugin SHARED
  native_plugin_abi.cpp
  msda_plugin.cu
  msda_creator.cpp
  layer_norm_plugin.cu
  layer_norm_creator.cpp
  sigmoid_plugin.cu
  sigmoid_creator.cpp
  softmax_plugin.cu
  softmax_creator.cpp
  mha_scale_plugin.cu
  mha_scale_creator.cpp
  hiera_patch_conv_plugin.cu
  hiera_patch_conv_creator.cpp
  hiera_layer_norm_plugin.cu
  hiera_layer_norm_creator.cpp
  hiera_flash_attention_launcher.cu
  hiera_flash_attention_plugin.cpp
  hiera_flash_attention_creator.cpp
  hiera_gelu_launcher.cu
  hiera_gelu_plugin.cpp
  hiera_gelu_creator.cpp
  hiera_block1415_projection_plugin.cpp
  hiera_block1415_projection_creator.cpp
)
target_include_directories(trtmc_sam2_hoi_native_plugin BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/flash_attention/include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cutlass/include
)
target_include_directories(trtmc_sam2_hoi_native_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SAM2_HOI_TRT_INCLUDE_DIR}
)
target_link_libraries(trtmc_sam2_hoi_native_plugin PRIVATE
  ${SAM2_HOI_TRT_LIBRARY}
  CUDA::cudart
  CUDA::cublasLt
)
target_compile_options(trtmc_sam2_hoi_native_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
)
# Match the reviewed PyTorch 2.7.1 FlashAttention compilation boundary without
# changing CUDA flags for any other SAM2-HOI kernel in this family DSO.  Do not
# add --use_fast_math: it changes reciprocal normalization and breaks BF16
# bit-exactness on both CUDA 12.8 and CUDA 13.3.
set_source_files_properties(hiera_flash_attention_launcher.cu PROPERTIES
  COMPILE_DEFINITIONS "UNFUSE_FMA=1"
  COMPILE_OPTIONS
    "-O2;--expt-relaxed-constexpr;--expt-extended-lambda;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__"
)
# The current qualification targets are NVIDIA L4 (SM89) and GB300 (SM100).
set_property(TARGET trtmc_sam2_hoi_native_plugin PROPERTY CUDA_ARCHITECTURES 89 100)

get_filename_component(SAM2_HOI_TRT_LIBRARY_DIR "${SAM2_HOI_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_sam2_hoi_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_sam2_hoi_native_plugin
  BUILD_RPATH "${SAM2_HOI_TRT_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
