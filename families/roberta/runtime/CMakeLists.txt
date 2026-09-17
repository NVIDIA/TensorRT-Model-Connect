# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_roberta SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  unigram_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_roberta
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_roberta SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_roberta PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_roberta PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_roberta PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_roberta
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_roberta_added_token_whitespace
    ${PROJECT_SOURCE_DIR}/families/roberta/tests/cpp/test_roberta_added_token_whitespace.cpp
    bpe_tokenizer.cpp
  )
  target_include_directories(test_roberta_added_token_whitespace PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_roberta_added_token_whitespace PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_roberta_added_token_whitespace PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME roberta_added_token_whitespace COMMAND test_roberta_added_token_whitespace)
endif()
