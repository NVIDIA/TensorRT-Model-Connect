# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(trtmc_audio_streaming_example LANGUAGES CXX)

find_package(trtmc CONFIG REQUIRED)

add_executable(trtmc_audio_streaming main.cpp)
target_compile_features(trtmc_audio_streaming PRIVATE cxx_std_17)
target_compile_options(trtmc_audio_streaming PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(trtmc_audio_streaming PRIVATE trtmc::trtmc_runtime)
