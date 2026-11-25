# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(h3_video_benchmark LANGUAGES CXX)
if(NOT WIN32)
  message(FATAL_ERROR "This consumer uses the native Windows media implementation")
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
set(TRTMC_RUNTIME_LIBRARY "" CACHE FILEPATH "Existing ModelConnect runtime import library")
if(NOT EXISTS "${TRTMC_RUNTIME_LIBRARY}")
  message(FATAL_ERROR "Set TRTMC_RUNTIME_LIBRARY to the existing trtmc_runtime.lib")
endif()
find_package(nlohmann_json 3.11 REQUIRED)
get_filename_component(_repo "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(h3_benchmark_video
  benchmark_video.cpp
  "${_repo}/apps/cli/io.cpp"
  "${_repo}/apps/cli/windows_media.cpp"
  "${_repo}/apps/cli/trtmc_windows_utf8.manifest"
)
target_compile_features(h3_benchmark_video PRIVATE cxx_std_17)
target_compile_options(h3_benchmark_video PRIVATE /utf-8)
target_include_directories(h3_benchmark_video PRIVATE
  "${_repo}/core/runtime/include" "${_repo}/apps" "${_repo}/third_party/stb")
target_link_libraries(h3_benchmark_video PRIVATE
  "${TRTMC_RUNTIME_LIBRARY}" nlohmann_json::nlohmann_json mfplat mfreadwrite mfuuid ole32)
