# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_olmo SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_olmo
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_olmo SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_olmo PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_olmo PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_olmo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_olmo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_olmo_prompt_tokenization
    ${PROJECT_SOURCE_DIR}/families/olmo/tests/cpp/test_olmo_prompt_tokenization.cpp)
  target_include_directories(test_olmo_prompt_tokenization PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_olmo_prompt_tokenization PRIVATE trtmc_model_olmo nlohmann_json::nlohmann_json)
  add_test(NAME olmo_prompt_tokenization COMMAND test_olmo_prompt_tokenization)
  set_tests_properties(olmo_prompt_tokenization PROPERTIES LABELS "cpu")
endif()
