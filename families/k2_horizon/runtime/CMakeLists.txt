# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_k2_horizon SHARED
  bpe_tokenizer.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_k2_horizon
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_k2_horizon SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_k2_horizon PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_k2_horizon PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_k2_horizon PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_k2_horizon
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_k2_horizon_sampler
    ${PROJECT_SOURCE_DIR}/families/k2_horizon/tests/cpp/test_k2_horizon_sampler.cpp
  )
  target_include_directories(test_k2_horizon_sampler PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_k2_horizon_sampler PRIVATE
    trtmc_model_k2_horizon
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME test_k2_horizon_sampler COMMAND test_k2_horizon_sampler)
endif()
