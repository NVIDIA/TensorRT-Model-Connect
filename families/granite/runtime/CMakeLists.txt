# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_granite SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_granite
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_granite SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_granite PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_granite PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_granite PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_granite
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # CPU-backed KV-cache test: CUDA calls are served from host memory, and an
  # overlapping copy fails the test.
  add_executable(test_granite_kv_cache
    ${PROJECT_SOURCE_DIR}/families/granite/tests/cpp/test_kv_cache.cpp
    kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
  )
  target_include_directories(test_granite_kv_cache PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_granite_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_granite_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME granite_kv_cache COMMAND test_granite_kv_cache)
endif()
