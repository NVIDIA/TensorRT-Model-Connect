# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_wan22_dit_cuda_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_path(WAN22_DIT_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
)
find_library(WAN22_DIT_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_DIT_TRT_INCLUDE_DIR OR NOT WAN22_DIT_TRT_LIBRARY)
  message(FATAL_ERROR "Wan2.2 DiT CUDA plugin requires TensorRT headers and libnvinfer")
endif()

find_path(WAN22_DIT_CUDNN_INCLUDE_DIR cudnn.h
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
)
find_library(WAN22_DIT_CUDNN_LIBRARY
  NAMES cudnn libcudnn.so.9 libcudnn.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_DIT_CUDNN_INCLUDE_DIR OR NOT WAN22_DIT_CUDNN_LIBRARY)
  message(FATAL_ERROR "Wan2.2 DiT CUDA plugins require cuDNN headers and libcudnn")
endif()

set(WAN22_DIT_CUDNN_FRONTEND_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn_frontend/include"
)
if(NOT EXISTS "${WAN22_DIT_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
  message(FATAL_ERROR "Wan2.2 DiT SDPA requires the vendored cuDNN frontend 1.22.1 headers")
endif()

add_library(trtmc_wan22_dit_cuda_plugin SHARED
  wan22_adaptive_norm_fp32_plugin.cu
  wan22_bf16_linear_plugin.cu
  wan22_dit_numeric_plugins.cu
  wan22_final_projection_fp32_plugin.cu
  wan22_gated_residual_fp32_plugin.cu
  wan22_cudnn_sdpa_plugin.cpp
  wan22_layer_norm_fp32_plugin.cu
  wan22_patch_embedding_plugin.cu
  wan22_rms_norm_fp32_plugin.cu
  wan22_silu_fp32_plugin.cu
  wan22_time_linear1_plugin.cu
  wan22_time_linear2_plugin.cu
  wan22_time_projection_plugin.cu
)
target_compile_definitions(trtmc_wan22_dit_cuda_plugin PRIVATE
  CUDNN_FRONTEND_SKIP_JSON_LIB
)
target_include_directories(trtmc_wan22_dit_cuda_plugin PRIVATE
  ${WAN22_DIT_TRT_INCLUDE_DIR}
  ${WAN22_DIT_CUDNN_INCLUDE_DIR}
  ${WAN22_DIT_CUDNN_FRONTEND_INCLUDE_DIR}
)
target_link_libraries(trtmc_wan22_dit_cuda_plugin PRIVATE
  ${WAN22_DIT_TRT_LIBRARY}
  ${WAN22_DIT_CUDNN_LIBRARY}
  CUDA::cublasLt
  CUDA::cudart
  CUDA::nvrtc
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_wan22_dit_cuda_plugin PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=true>"
)

# GB300 is SM103 and Jetson Thor is SM110. Package real cubins for both
# qualification targets, so Model-Connect plugin source is never compiled at
# runtime. The cuDNN frontend SDPA path retains its separate NVRTC dependency.
set_property(TARGET trtmc_wan22_dit_cuda_plugin PROPERTY CUDA_ARCHITECTURES "103;110")

get_filename_component(WAN22_DIT_TRT_LIBRARY_DIR "${WAN22_DIT_TRT_LIBRARY}" DIRECTORY)
get_filename_component(WAN22_DIT_CUDNN_LIBRARY_DIR "${WAN22_DIT_CUDNN_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_wan22_dit_cuda_plugin PROPERTIES
  OUTPUT_NAME trtmc_wan22_dit_cuda_plugin
  BUILD_RPATH
    "${WAN22_DIT_TRT_LIBRARY_DIR};${WAN22_DIT_CUDNN_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
