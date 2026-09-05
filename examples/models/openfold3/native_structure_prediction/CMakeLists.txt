# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(trtmc_openfold3_native_structure_prediction LANGUAGES CXX)

get_filename_component(_trtmc_default_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(TRTMC_SOURCE_DIR "${_trtmc_default_source_dir}" CACHE PATH
    "TensorRT-Model-Connect source directory")
if(NOT EXISTS "${TRTMC_SOURCE_DIR}/include/trtmc/pipeline.h")
  message(FATAL_ERROR "TRTMC_SOURCE_DIR is not a TensorRT-Model-Connect source tree")
endif()

set(TRTMC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL OFF CACHE BOOL "" FORCE)
set(TRTMC_ENABLE_TVM_FFI OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BACKEND_RTX OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_DIFFUSION_KERNELS OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BACKEND_TRT ON CACHE BOOL "" FORCE)

add_subdirectory("${TRTMC_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/trtmc")

add_executable(trtmc_openfold3_native main.cpp)
target_compile_features(trtmc_openfold3_native PRIVATE cxx_std_17)
target_compile_options(trtmc_openfold3_native PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(trtmc_openfold3_native PRIVATE trtmc_core)
add_dependencies(trtmc_openfold3_native trtmc_model_openfold3 trtmc_backend_trt)
set_target_properties(trtmc_openfold3_native PROPERTIES
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/trtmc"
  BUILD_RPATH_USE_ORIGIN TRUE
)
