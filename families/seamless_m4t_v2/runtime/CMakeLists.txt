# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_seamless_m4t_v2 SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_seamless_m4t_v2 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_seamless_m4t_v2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_seamless_m4t_v2 PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n)
target_compile_options(trtmc_model_seamless_m4t_v2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_seamless_m4t_v2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_seamless_m4t_v2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_seamless_m4t_v2_tokenizer
    ${PROJECT_SOURCE_DIR}/families/seamless_m4t_v2/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_seamless_m4t_v2_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_seamless_m4t_v2_tokenizer PRIVATE
    trtmc_model_seamless_m4t_v2 nlohmann_json::nlohmann_json)
  set_target_properties(test_seamless_m4t_v2_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
