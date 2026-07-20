# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(trtmc_qwen_edgellm_coexistence LANGUAGES CXX)

set(TRTMC_MC_INCLUDE_DIR "" CACHE PATH "Directory containing trtmc/pipeline.h")
set(TRTMC_MC_CORE_LIBRARY "" CACHE FILEPATH "Model Connect core library")

if(NOT EXISTS "${TRTMC_MC_INCLUDE_DIR}/trtmc/pipeline.h")
  message(FATAL_ERROR "TRTMC_MC_INCLUDE_DIR must contain trtmc/pipeline.h")
endif()
if(NOT EXISTS "${TRTMC_MC_CORE_LIBRARY}")
  message(FATAL_ERROR "TRTMC_MC_CORE_LIBRARY must name the built Model Connect core library")
endif()

find_package(Threads REQUIRED)

add_executable(trtmc_qwen_edgellm_coexistence_runner coexistence_runner.cpp)
target_compile_features(trtmc_qwen_edgellm_coexistence_runner PRIVATE cxx_std_17)
target_compile_options(trtmc_qwen_edgellm_coexistence_runner PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)
target_include_directories(trtmc_qwen_edgellm_coexistence_runner PRIVATE
  "${TRTMC_MC_INCLUDE_DIR}"
)
target_link_libraries(trtmc_qwen_edgellm_coexistence_runner PRIVATE
  "${TRTMC_MC_CORE_LIBRARY}"
  Threads::Threads
)
