# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_qwen3_8_native_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(QWEN3_8_TRT_INCLUDE_DIR "" CACHE PATH "Directory with NvInferRuntime.h")
set(QWEN3_8_TRT_LIBRARY "" CACHE FILEPATH "Path to libnvinfer.so.11")
set(QWEN3_8_CUDA_INCLUDE_DIR "" CACHE PATH "Directory with cuda.h and cuda_runtime_api.h")

# Only the driver API is called; libcuda comes from the installed GPU driver.
find_library(QWEN3_8_CUDA_DRIVER_LIBRARY NAMES cuda REQUIRED
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64
        /usr/local/cuda/lib64/stubs)

add_library(trtmc_qwen3_8_native_plugin SHARED gdn_tilegym_plugin.cpp
  gdn_prefill_plugin.cpp)
target_include_directories(trtmc_qwen3_8_native_plugin PRIVATE
  ${QWEN3_8_TRT_INCLUDE_DIR}
  ${QWEN3_8_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_qwen3_8_native_plugin PRIVATE
  ${QWEN3_8_TRT_LIBRARY}
  ${QWEN3_8_CUDA_DRIVER_LIBRARY}
)
target_compile_options(trtmc_qwen3_8_native_plugin PRIVATE -O3 -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_qwen3_8_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_qwen3_8_native_plugin
)
