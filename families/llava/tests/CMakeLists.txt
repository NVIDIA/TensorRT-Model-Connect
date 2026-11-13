# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_llava_kv_cache cpp/test_kv_cache.cpp
  ../runtime/kv_cache.cpp ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp)
target_include_directories(test_llava_kv_cache PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(test_llava_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_compile_options(test_llava_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(test_llava_kv_cache PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(mode zero exhaustion allocation)
  add_test(NAME test_llava_kv_cache_${mode} COMMAND test_llava_kv_cache ${mode})
endforeach()
