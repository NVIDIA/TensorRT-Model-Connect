# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_phi_moe SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_phi_moe
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_phi_moe SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_phi_moe PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_phi_moe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_phi_moe PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_phi_moe
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_phi_moe_tokenizer
    ${PROJECT_SOURCE_DIR}/families/phi_moe/tests/cpp/test_phi_moe_tokenizer.cpp)
  target_include_directories(test_phi_moe_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_phi_moe_tokenizer PRIVATE trtmc_model_phi_moe)
  add_test(NAME phi_moe_tokenizer COMMAND test_phi_moe_tokenizer)
  set_tests_properties(phi_moe_tokenizer PROPERTIES LABELS "cpu")
endif()
