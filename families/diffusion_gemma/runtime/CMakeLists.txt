# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_diffusion_gemma SHARED
  bpe_tokenizer.cpp generation.cpp pipeline.cpp plugin.cpp sampling.cpp)
target_include_directories(trtmc_model_diffusion_gemma PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_diffusion_gemma SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_diffusion_gemma PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_diffusion_gemma PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_diffusion_gemma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_diffusion_gemma LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_diffusion_gemma_tokenizer
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_diffusion_gemma_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_diffusion_gemma_tokenizer PRIVATE trtmc_model_diffusion_gemma nlohmann_json::nlohmann_json)
  add_test(NAME test_diffusion_gemma_tokenizer COMMAND test_diffusion_gemma_tokenizer)
  set_target_properties(test_diffusion_gemma_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_diffusion_gemma_sampling
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_sampling.cpp)
  target_include_directories(test_diffusion_gemma_sampling PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_diffusion_gemma_sampling SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_diffusion_gemma_sampling PRIVATE trtmc_model_diffusion_gemma)
  add_test(NAME test_diffusion_gemma_sampling COMMAND test_diffusion_gemma_sampling)

  add_executable(test_diffusion_gemma_pipeline
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_pipeline.cpp)
  target_include_directories(test_diffusion_gemma_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_diffusion_gemma_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_diffusion_gemma_pipeline PRIVATE
    trtmc_model_diffusion_gemma trtmc_core nlohmann_json::nlohmann_json)
  add_test(NAME test_diffusion_gemma_pipeline COMMAND test_diffusion_gemma_pipeline)

  add_executable(test_diffusion_gemma_diagnostic
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/diagnostic.cpp)
  target_include_directories(test_diffusion_gemma_diagnostic PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_diffusion_gemma_diagnostic SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_diffusion_gemma_diagnostic PRIVATE
    trtmc_model_diffusion_gemma trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
  set_target_properties(test_diffusion_gemma_sampling test_diffusion_gemma_pipeline
    test_diffusion_gemma_diagnostic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_diffusion_gemma_sdk_cpp
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/sdk_consumer.cpp)
  target_link_libraries(test_diffusion_gemma_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_diffusion_gemma_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
