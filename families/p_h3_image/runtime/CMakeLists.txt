# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_p_h3_image SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_p_h3_image PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_p_h3_image SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_p_h3_image PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n)
target_compile_options(trtmc_model_p_h3_image PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_p_h3_image PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_p_h3_image LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_p_h3_image_runtime ${PROJECT_SOURCE_DIR}/families/p_h3_image/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_p_h3_image_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_p_h3_image_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_p_h3_image_tokenizer ${PROJECT_SOURCE_DIR}/families/p_h3_image/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_p_h3_image_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_p_h3_image_tokenizer PRIVATE trtmc_model_p_h3_image nlohmann_json::nlohmann_json)
  set_target_properties(test_p_h3_image_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

