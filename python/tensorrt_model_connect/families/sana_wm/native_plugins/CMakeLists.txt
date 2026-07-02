# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_sana_wm_native_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

find_path(SANA_WM_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(SANA_WM_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT SANA_WM_TRT_INCLUDE_DIR OR NOT SANA_WM_TRT_LIBRARY)
  message(FATAL_ERROR "SANA-WM native plugin requires TensorRT headers and libnvinfer")
endif()

add_library(trtmc_sana_wm_native_plugin SHARED
  sana_wm_gdn_plugin.cu
  sana_wm_gdn_creator.cpp
  sana_wm_torch_conv2d_plugin.cpp
  torch_cuda_bfloat16_math.cpp
)

target_include_directories(trtmc_sana_wm_native_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SANA_WM_TRT_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(trtmc_sana_wm_native_plugin PRIVATE
  ${SANA_WM_TRT_LIBRARY}
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
  ${TORCH_LIBRARIES}
)
target_compile_definitions(trtmc_sana_wm_native_plugin PRIVATE
  TRTMC_HAS_TRT=1
  TRTMC_HAS_CUDNN=0
  TRTMC_HAS_LIBTORCH_CONV=1
  TRTMC_HAS_LIBTORCH_MULTINOMIAL=1
  TRTMC_HAS_CUDA_KERNELS=1
)
target_compile_options(trtmc_sana_wm_native_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
)
set_property(TARGET trtmc_sana_wm_native_plugin PROPERTY CUDA_ARCHITECTURES 75)

get_filename_component(SANA_WM_TRT_LIBRARY_DIR "${SANA_WM_TRT_LIBRARY}" DIRECTORY)
get_filename_component(SANA_WM_TORCH_LIBRARY_DIR "${TORCH_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_sana_wm_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_sana_wm_native_plugin
  BUILD_RPATH "${SANA_WM_TRT_LIBRARY_DIR};${SANA_WM_TORCH_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
