# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nomic_bert SHARED pipeline.cpp plugin.cpp wordpiece_tokenizer.cpp)
target_include_directories(trtmc_model_nomic_bert PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_nomic_bert SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_nomic_bert PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_nomic_bert PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_nomic_bert PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_nomic_bert LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_nomic_bert_tokenizer
    ${PROJECT_SOURCE_DIR}/families/nomic_bert/tests/cpp/test_tokenizer.cpp wordpiece_tokenizer.cpp)
  target_include_directories(test_nomic_bert_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_nomic_bert_tokenizer PRIVATE nlohmann_json::nlohmann_json)
  add_test(NAME nomic_bert_tokenizer COMMAND test_nomic_bert_tokenizer)
  set_tests_properties(nomic_bert_tokenizer PROPERTIES LABELS cpu)
  add_executable(nomic_embedding_consumer
    ${PROJECT_SOURCE_DIR}/families/nomic_bert/tests/cpp/embedding_consumer.cpp)
  target_link_libraries(nomic_embedding_consumer PRIVATE trtmc_c)
endif()
