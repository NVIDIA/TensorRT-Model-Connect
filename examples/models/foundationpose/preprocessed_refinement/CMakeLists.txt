# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(trtmc_foundationpose_preprocessed LANGUAGES CXX)

get_filename_component(_trtmc_default_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(TRTMC_SOURCE_DIR "${_trtmc_default_source_dir}" CACHE PATH
    "TensorRT-Model-Connect source directory")
if(NOT EXISTS "${TRTMC_SOURCE_DIR}/core/runtime/include/trtmc/task.h")
  message(FATAL_ERROR "TRTMC_SOURCE_DIR is not a TensorRT-Model-Connect source tree")
endif()

set(TRTMC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TRTMC_ENABLE_BYOK OFF CACHE BOOL "" FORCE)
add_subdirectory("${TRTMC_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/trtmc")

add_executable(trtmc_foundationpose_preprocessed main.cpp)
target_compile_features(trtmc_foundationpose_preprocessed PRIVATE cxx_std_17)
target_compile_options(trtmc_foundationpose_preprocessed PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(trtmc_foundationpose_preprocessed PRIVATE
  "${TRTMC_SOURCE_DIR}/core/runtime/include"
)
target_link_libraries(trtmc_foundationpose_preprocessed PRIVATE trtmc_runtime)
add_dependencies(trtmc_foundationpose_preprocessed trtmc_model_foundationpose trtmc_backend_trt)
set_target_properties(trtmc_foundationpose_preprocessed PROPERTIES
  BUILD_RPATH "\$ORIGIN;\$ORIGIN/trtmc"
  BUILD_RPATH_USE_ORIGIN TRUE
)
