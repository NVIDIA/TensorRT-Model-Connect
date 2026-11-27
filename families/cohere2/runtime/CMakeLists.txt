# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_cohere2 SHARED model.cpp plugin.cpp tokenizer.cpp)
target_include_directories(trtmc_model_cohere2 PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_cohere2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_cohere2 PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_cohere2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_cohere2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_cohere2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_cohere2_tokenizer ${PROJECT_SOURCE_DIR}/families/cohere2/tests/cpp/test_tokenizer.cpp tokenizer.cpp)
  target_include_directories(test_cohere2_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_cohere2_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  set_target_properties(test_cohere2_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_cohere2_tokenizer COMMAND test_cohere2_tokenizer)
endif()
