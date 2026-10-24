# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_embeddinggemma2 SHARED bpe_tokenizer.cpp pipeline.cpp plugin.cpp)
target_include_directories(trtmc_model_embeddinggemma2 PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_embeddinggemma2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_embeddinggemma2 PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_embeddinggemma2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_embeddinggemma2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_embeddinggemma2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_embeddinggemma2_pipeline
    ${PROJECT_SOURCE_DIR}/families/embeddinggemma2/tests/cpp/test_pipeline.cpp)
  target_include_directories(test_embeddinggemma2_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_embeddinggemma2_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_embeddinggemma2_pipeline PRIVATE trtmc_model_embeddinggemma2)
  add_test(NAME test_embeddinggemma2_pipeline COMMAND test_embeddinggemma2_pipeline)
  add_executable(test_embeddinggemma2_tokenizer
    ${PROJECT_SOURCE_DIR}/families/embeddinggemma2/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_embeddinggemma2_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_embeddinggemma2_tokenizer PRIVATE trtmc_model_embeddinggemma2)
  add_test(NAME test_embeddinggemma2_tokenizer COMMAND test_embeddinggemma2_tokenizer)
  add_executable(embeddinggemma2_consumer
    ${PROJECT_SOURCE_DIR}/families/embeddinggemma2/tests/cpp/consumer.cpp)
  target_link_libraries(embeddinggemma2_consumer PRIVATE trtmc_c)
endif()
