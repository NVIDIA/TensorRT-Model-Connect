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

find_path(SANA_WM_TRT_INCLUDE_DIR
  NAMES NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
  REQUIRED
)
find_library(SANA_WM_TRT_LIBRARY
  NAMES libnvinfer.so.11
  PATHS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
  REQUIRED
)

add_library(trtmc_sana_wm_native_plugin SHARED
  sana_wm_gdn_plugin.cu
  sana_wm_gdn_creator.cpp
  sana_wm_torch_conv2d_plugin.cpp
)
set_source_files_properties(sana_wm_gdn_creator.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-deprecated-declarations"
)
set_source_files_properties(sana_wm_gdn_plugin.cu PROPERTIES
  COMPILE_OPTIONS "-diag-suppress=997"
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
target_compile_options(trtmc_sana_wm_native_plugin PRIVATE
  ${TORCH_CXX_FLAGS}
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Wall;-Wextra>"
)
set_property(TARGET trtmc_sana_wm_native_plugin PROPERTY CUDA_ARCHITECTURES 75)

get_filename_component(SANA_WM_TRT_LIBRARY_DIR "${SANA_WM_TRT_LIBRARY}" DIRECTORY)
get_filename_component(SANA_WM_TORCH_LIBRARY_DIR "${TORCH_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_sana_wm_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_sana_wm_native_plugin
  BUILD_RPATH "${SANA_WM_TRT_LIBRARY_DIR};${SANA_WM_TORCH_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
