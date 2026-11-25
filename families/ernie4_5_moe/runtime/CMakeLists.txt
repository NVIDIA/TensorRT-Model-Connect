# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_ernie4_5_moe SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_ernie4_5_moe PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_ernie4_5_moe SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_ernie4_5_moe PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_ernie4_5_moe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_ernie4_5_moe PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_ernie4_5_moe LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_ernie4_5_moe_tokenizer ${PROJECT_SOURCE_DIR}/families/ernie4_5_moe/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_ernie4_5_moe_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_ernie4_5_moe_tokenizer PRIVATE trtmc_model_ernie4_5_moe nlohmann_json::nlohmann_json)
  set_target_properties(test_ernie4_5_moe_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_ernie4_5_moe_tokenizer COMMAND test_ernie4_5_moe_tokenizer)
  add_executable(test_ernie4_5_moe_runtime ${PROJECT_SOURCE_DIR}/families/ernie4_5_moe/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_ernie4_5_moe_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_ernie4_5_moe_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
