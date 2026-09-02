# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(trtmc_voicechat_full_duplex_example LANGUAGES CXX)

get_filename_component(_trtmc_default_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(TRTMC_SOURCE_DIR "${_trtmc_default_source_dir}" CACHE PATH
    "TensorRT-Model-Connect source directory")
if(NOT EXISTS "${TRTMC_SOURCE_DIR}/include/trtmc/speech_session.h")
  message(FATAL_ERROR "TRTMC_SOURCE_DIR is not a TensorRT-Model-Connect source tree")
endif()

# Keep this example build focused on the runtime, the VoiceChat DSO, and this
# executable. The Docker build selects the concrete targets it needs.
set(TRTMC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL OFF CACHE BOOL "" FORCE)
set(TRTMC_ENABLE_TVM_FFI OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BACKEND_RTX OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_DIFFUSION_KERNELS OFF CACHE BOOL "" FORCE)
set(TRTMC_BUILD_BACKEND_TRT ON CACHE BOOL "" FORCE)

find_package(ALSA REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory("${TRTMC_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/trtmc")

add_executable(trtmc_voicechat_full_duplex main.cpp)
target_compile_features(trtmc_voicechat_full_duplex PRIVATE cxx_std_17)
target_compile_options(trtmc_voicechat_full_duplex PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(trtmc_voicechat_full_duplex PRIVATE
  trtmc_core
  ALSA::ALSA
  Threads::Threads
)
set_target_properties(trtmc_voicechat_full_duplex PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  BUILD_RPATH_USE_ORIGIN TRUE
)

option(TRTMC_VOICECHAT_BUILD_QUEUE_TEST "Build the device-free playback queue test" ON)
if(TRTMC_VOICECHAT_BUILD_QUEUE_TEST)
  enable_testing()
  add_executable(test_trtmc_voicechat_playback_queue test_playback_queue.cpp)
  target_compile_features(test_trtmc_voicechat_playback_queue PRIVATE cxx_std_17)
  target_compile_options(test_trtmc_voicechat_playback_queue PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(test_trtmc_voicechat_playback_queue PRIVATE Threads::Threads)
  add_test(NAME trtmc_voicechat_playback_queue COMMAND test_trtmc_voicechat_playback_queue)
endif()
