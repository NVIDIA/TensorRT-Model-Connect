# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_musicgen SHARED plugin.cpp unigram_tokenizer.cpp)
target_include_directories(trtmc_model_musicgen PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(trtmc_model_musicgen PRIVATE trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_musicgen PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_musicgen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_musicgen LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_musicgen_tokenizer_config ../tests/test_tokenizer_config.cpp unigram_tokenizer.cpp)
  target_include_directories(test_musicgen_tokenizer_config PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_musicgen_tokenizer_config PRIVATE nlohmann_json::nlohmann_json)
  add_test(NAME musicgen_tokenizer_config COMMAND test_musicgen_tokenizer_config)
  add_executable(test_musicgen_sdk_cpp ${PROJECT_SOURCE_DIR}/families/musicgen/tests/sdk_consumer.cpp)
  target_link_libraries(test_musicgen_sdk_cpp PRIVATE trtmc_c)
  target_compile_options(test_musicgen_sdk_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_musicgen_sdk_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_dependencies(trtmc_model_musicgen test_musicgen_sdk_cpp)
  add_executable(test_musicgen_tokenizer ${PROJECT_SOURCE_DIR}/families/musicgen/tests/tokenizer_consumer.cpp)
  target_include_directories(test_musicgen_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_musicgen_tokenizer PRIVATE trtmc_model_musicgen nlohmann_json::nlohmann_json)
  target_compile_options(test_musicgen_tokenizer PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_musicgen_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
