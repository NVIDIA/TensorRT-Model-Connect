# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_wan22_vae_cuda_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

find_path(WAN22_VAE_CUDNN_INCLUDE_DIR cudnn.h
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/cuda/include
)
find_library(WAN22_VAE_CUDNN_LIBRARY
  NAMES cudnn libcudnn.so.9 libcudnn.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/cuda/lib64
)
if(NOT WAN22_VAE_CUDNN_INCLUDE_DIR OR NOT WAN22_VAE_CUDNN_LIBRARY)
  message(FATAL_ERROR "Wan2.2 VAE Conv3d plugin requires cuDNN headers and libcudnn")
endif()

find_path(WAN22_VAE_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)
find_library(WAN22_VAE_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_VAE_TRT_INCLUDE_DIR OR NOT WAN22_VAE_TRT_LIBRARY)
  message(FATAL_ERROR "Wan2.2 VAE CUDA plugin requires TensorRT headers and libnvinfer")
endif()

add_library(trtmc_wan22_vae_cuda_plugin SHARED
  wan22_vae_fp32_barrier_plugin.cu
  wan22_vae_rms_norm_plugin.cu
  wan22_vae_conv3d_plugin.cu
)
target_include_directories(trtmc_wan22_vae_cuda_plugin PRIVATE
  ${WAN22_VAE_TRT_INCLUDE_DIR}
  ${WAN22_VAE_CUDNN_INCLUDE_DIR}
)
target_link_libraries(trtmc_wan22_vae_cuda_plugin PRIVATE
  ${WAN22_VAE_TRT_LIBRARY}
  ${WAN22_VAE_CUDNN_LIBRARY}
  CUDA::cudart
)
target_compile_options(trtmc_wan22_vae_cuda_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
)

# GB300 is SM103; Jetson Thor is SM110.
set_property(TARGET trtmc_wan22_vae_cuda_plugin PROPERTY CUDA_ARCHITECTURES "103;110")

get_filename_component(WAN22_VAE_TRT_LIBRARY_DIR "${WAN22_VAE_TRT_LIBRARY}" DIRECTORY)
get_filename_component(WAN22_VAE_CUDNN_LIBRARY_DIR "${WAN22_VAE_CUDNN_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_wan22_vae_cuda_plugin PROPERTIES
  OUTPUT_NAME trtmc_wan22_vae_cuda_plugin
  BUILD_RPATH "${WAN22_VAE_TRT_LIBRARY_DIR};${WAN22_VAE_CUDNN_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
