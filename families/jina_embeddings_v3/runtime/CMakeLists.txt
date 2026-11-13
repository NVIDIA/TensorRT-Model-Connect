# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_jina_embeddings_v3 SHARED unigram_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_jina_embeddings_v3 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_jina_embeddings_v3 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_jina_embeddings_v3 PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_jina_embeddings_v3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_jina_embeddings_v3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_jina_embeddings_v3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_jina_embeddings_v3_tokenizer
    ${PROJECT_SOURCE_DIR}/families/jina_embeddings_v3/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_jina_embeddings_v3_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_jina_embeddings_v3_tokenizer PRIVATE
    trtmc_model_jina_embeddings_v3 nlohmann_json::nlohmann_json)
  set_target_properties(test_jina_embeddings_v3_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
