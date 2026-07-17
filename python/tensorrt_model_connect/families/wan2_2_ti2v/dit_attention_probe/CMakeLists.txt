# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_wan22_cudnn_sdpa_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_path(WAN22_SDPA_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
)
find_library(WAN22_SDPA_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
set(WAN22_SDPA_CUDNN_FRONTEND_INCLUDE_DIR "" CACHE PATH
  "Path containing the exact cuDNN frontend 1.22.1 cudnn_frontend.h"
)
find_path(WAN22_SDPA_CUDNN_INCLUDE_DIR cudnn.h
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
)
find_library(WAN22_SDPA_CUDNN_LIBRARY
  NAMES cudnn libcudnn.so.9
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_SDPA_TRT_INCLUDE_DIR OR NOT WAN22_SDPA_TRT_LIBRARY)
  message(FATAL_ERROR "Wan2.2 SDPA probe requires TensorRT headers and libnvinfer")
endif()
if(NOT EXISTS "${WAN22_SDPA_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
  message(FATAL_ERROR "Set WAN22_SDPA_CUDNN_FRONTEND_INCLUDE_DIR to cuDNN frontend 1.22.1 include")
endif()
if(NOT WAN22_SDPA_CUDNN_INCLUDE_DIR OR NOT WAN22_SDPA_CUDNN_LIBRARY)
  message(FATAL_ERROR "Wan2.2 SDPA probe requires cuDNN headers and libcudnn")
endif()

add_library(trtmc_wan22_cudnn_sdpa_probe SHARED wan22_cudnn_sdpa_probe.cpp)
target_compile_definitions(trtmc_wan22_cudnn_sdpa_probe PRIVATE CUDNN_FRONTEND_SKIP_JSON_LIB)
target_include_directories(trtmc_wan22_cudnn_sdpa_probe PRIVATE
  ${WAN22_SDPA_TRT_INCLUDE_DIR}
  ${WAN22_SDPA_CUDNN_FRONTEND_INCLUDE_DIR}
  ${WAN22_SDPA_CUDNN_INCLUDE_DIR}
)
target_link_libraries(trtmc_wan22_cudnn_sdpa_probe PRIVATE
  ${WAN22_SDPA_TRT_LIBRARY}
  ${WAN22_SDPA_CUDNN_LIBRARY}
  CUDA::cudart
  CUDA::nvrtc
  dl
)
target_compile_options(trtmc_wan22_cudnn_sdpa_probe PRIVATE -O3 -Wall -Wextra -Wpedantic)

add_executable(wan22_cudnn_sdpa_trt_runner run_trt_probe.cpp)
target_include_directories(wan22_cudnn_sdpa_trt_runner PRIVATE ${WAN22_SDPA_TRT_INCLUDE_DIR})
target_link_libraries(wan22_cudnn_sdpa_trt_runner PRIVATE
  ${WAN22_SDPA_TRT_LIBRARY}
  CUDA::cudart
  dl
)
target_compile_options(wan22_cudnn_sdpa_trt_runner PRIVATE -O3 -Wall -Wextra -Wpedantic)

get_filename_component(WAN22_SDPA_TRT_LIBRARY_DIR "${WAN22_SDPA_TRT_LIBRARY}" DIRECTORY)
get_filename_component(WAN22_SDPA_CUDNN_LIBRARY_DIR "${WAN22_SDPA_CUDNN_LIBRARY}" DIRECTORY)
set(WAN22_SDPA_RPATH
  "${WAN22_SDPA_TRT_LIBRARY_DIR};${WAN22_SDPA_CUDNN_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
set_target_properties(trtmc_wan22_cudnn_sdpa_probe PROPERTIES
  OUTPUT_NAME trtmc_wan22_cudnn_sdpa_probe
  BUILD_RPATH "${WAN22_SDPA_RPATH}"
)
set_target_properties(wan22_cudnn_sdpa_trt_runner PROPERTIES BUILD_RPATH "${WAN22_SDPA_RPATH}")
