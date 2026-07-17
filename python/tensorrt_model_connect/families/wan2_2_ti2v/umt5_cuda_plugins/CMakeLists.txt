# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_wan22_umt5_cuda_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

find_path(WAN22_UMT5_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(WAN22_UMT5_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_UMT5_TRT_INCLUDE_DIR OR NOT WAN22_UMT5_TRT_LIBRARY)
  message(FATAL_ERROR "Wan2.2 UMT5 CUDA plugin requires TensorRT headers and libnvinfer")
endif()

add_library(trtmc_wan22_umt5_cuda_plugin SHARED
  wan22_umt5_gelu_plugin.cu
)
target_include_directories(trtmc_wan22_umt5_cuda_plugin PRIVATE
  ${WAN22_UMT5_TRT_INCLUDE_DIR}
)
target_link_libraries(trtmc_wan22_umt5_cuda_plugin PRIVATE
  ${WAN22_UMT5_TRT_LIBRARY}
  CUDA::cudart
)
target_compile_options(trtmc_wan22_umt5_cuda_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
)

# GB300 is SM103; Jetson Thor is SM110.  Keep both real cubins so the
# qualification artifact can be moved between the two target systems.
set_property(TARGET trtmc_wan22_umt5_cuda_plugin PROPERTY CUDA_ARCHITECTURES "103;110")

get_filename_component(WAN22_UMT5_TRT_LIBRARY_DIR "${WAN22_UMT5_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_wan22_umt5_cuda_plugin PROPERTIES
  OUTPUT_NAME trtmc_wan22_umt5_cuda_plugin
  BUILD_RPATH "${WAN22_UMT5_TRT_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
