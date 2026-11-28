# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_bailing_hybrid SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_bailing_hybrid PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_bailing_hybrid SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_bailing_hybrid PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n)
target_compile_options(trtmc_model_bailing_hybrid PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_bailing_hybrid PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_bailing_hybrid LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_bailing_hybrid_tokenizer ${PROJECT_SOURCE_DIR}/families/bailing_hybrid/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_bailing_hybrid_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_bailing_hybrid_tokenizer PRIVATE trtmc_model_bailing_hybrid nlohmann_json::nlohmann_json)
  set_target_properties(test_bailing_hybrid_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_bailing_hybrid_runtime ${PROJECT_SOURCE_DIR}/families/bailing_hybrid/tests/cpp/test_runtime.cpp)
  target_include_directories(test_bailing_hybrid_runtime PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_bailing_hybrid_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_bailing_hybrid_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_bailing_hybrid_runtime COMMAND test_bailing_hybrid_runtime)
  add_test(NAME test_bailing_hybrid_tokenizer COMMAND test_bailing_hybrid_tokenizer)
endif()
