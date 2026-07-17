# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_wan22_dit_linear_probe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_path(WAN22_LINEAR_TRT_INCLUDE_DIR NvInferRuntime.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
)
find_library(WAN22_LINEAR_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_LINEAR_TRT_INCLUDE_DIR OR NOT WAN22_LINEAR_TRT_LIBRARY)
  message(FATAL_ERROR "Wan2.2 DiT linear probe requires TensorRT headers and libnvinfer")
endif()

# cuDNN ships as a Python wheel in the qualification container.  Keep this
# lookup explicit so the experimental DSO links only the native runtime rather
# than accidentally inheriting a torch dependency.
find_path(WAN22_PATCH_CUDNN_INCLUDE_DIR cudnn.h
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
)
find_library(WAN22_PATCH_CUDNN_LIBRARY
  NAMES cudnn libcudnn.so.9 libcudnn.so
  PATHS
    /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(NOT WAN22_PATCH_CUDNN_INCLUDE_DIR OR NOT WAN22_PATCH_CUDNN_LIBRARY)
  message(FATAL_ERROR "Wan2.2 patch cuDNN probe requires cuDNN headers and libcudnn")
endif()

add_library(trtmc_wan22_dit_linear_probe SHARED wan22_dit_linear_probe_plugin.cu)
target_include_directories(trtmc_wan22_dit_linear_probe PRIVATE ${WAN22_LINEAR_TRT_INCLUDE_DIR})
target_link_libraries(trtmc_wan22_dit_linear_probe PRIVATE
  ${WAN22_LINEAR_TRT_LIBRARY}
  CUDA::cublasLt
  CUDA::cudart
)
target_compile_options(trtmc_wan22_dit_linear_probe PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra,-Wpedantic>"
)

# GB300 is SM103 and Jetson Thor is SM110.  This experimental DSO deliberately
# has no torch/ATen dependency and can be moved between the two qualification
# targets without runtime compilation.
set_property(TARGET trtmc_wan22_dit_linear_probe PROPERTY CUDA_ARCHITECTURES "103;110")

get_filename_component(WAN22_LINEAR_TRT_LIBRARY_DIR "${WAN22_LINEAR_TRT_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_wan22_dit_linear_probe PROPERTIES
  OUTPUT_NAME trtmc_wan22_dit_linear_probe
  BUILD_RPATH "${WAN22_LINEAR_TRT_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)

add_library(trtmc_wan22_final_projection_fp32_probe SHARED
  wan22_final_projection_fp32_probe.cu
)
target_link_libraries(trtmc_wan22_final_projection_fp32_probe PRIVATE
  CUDA::cublasLt
  CUDA::cudart
)
target_compile_options(trtmc_wan22_final_projection_fp32_probe PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=true>"
)
set_property(TARGET trtmc_wan22_final_projection_fp32_probe PROPERTY
  CUDA_ARCHITECTURES "103;110"
)
set_target_properties(trtmc_wan22_final_projection_fp32_probe PROPERTIES
  OUTPUT_NAME trtmc_wan22_final_projection_fp32_probe
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
)

add_library(trtmc_wan22_patch_cudnn_probe SHARED wan22_patch_cudnn_probe_plugin.cu)
target_include_directories(trtmc_wan22_patch_cudnn_probe PRIVATE
  ${WAN22_LINEAR_TRT_INCLUDE_DIR}
  ${WAN22_PATCH_CUDNN_INCLUDE_DIR}
)
target_link_libraries(trtmc_wan22_patch_cudnn_probe PRIVATE
  ${WAN22_LINEAR_TRT_LIBRARY}
  ${WAN22_PATCH_CUDNN_LIBRARY}
  CUDA::cudart
)
target_compile_options(trtmc_wan22_patch_cudnn_probe PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra,-Wpedantic>"
)
set_property(TARGET trtmc_wan22_patch_cudnn_probe PROPERTY CUDA_ARCHITECTURES "103;110")

get_filename_component(WAN22_PATCH_CUDNN_LIBRARY_DIR "${WAN22_PATCH_CUDNN_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_wan22_patch_cudnn_probe PROPERTIES
  OUTPUT_NAME trtmc_wan22_patch_cudnn_probe
  BUILD_RPATH
    "${WAN22_LINEAR_TRT_LIBRARY_DIR};${WAN22_PATCH_CUDNN_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
