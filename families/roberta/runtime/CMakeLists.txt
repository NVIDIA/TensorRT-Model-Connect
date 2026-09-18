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

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_roberta SHARED cli.cpp)
target_include_directories(trtmc_cli_roberta PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_roberta PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_roberta PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_roberta PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_roberta LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_roberta_cli
    ${PROJECT_SOURCE_DIR}/families/roberta/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_roberta_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_roberta_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_roberta_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME roberta_cli COMMAND test_roberta_cli)
  set_tests_properties(roberta_cli PROPERTIES LABELS cpu)
endif()
