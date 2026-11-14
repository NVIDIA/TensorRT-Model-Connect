# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_apertus SHARED
  kv_cache.cpp
  model.cpp
  plugin.cpp
  sampler.cpp
  tokenizer.cpp
)
target_include_directories(trtmc_model_apertus PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_apertus SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_apertus PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_apertus PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_apertus PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_apertus LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_apertus_tokenizer
    ${PROJECT_SOURCE_DIR}/families/apertus/tests/cpp/test_apertus_tokenizer.cpp tokenizer.cpp)
  target_include_directories(test_apertus_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_apertus_tokenizer PRIVATE
    nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  target_compile_options(test_apertus_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_apertus_tokenizer COMMAND test_apertus_tokenizer)

  add_executable(test_apertus_sampler
    ${PROJECT_SOURCE_DIR}/families/apertus/tests/cpp/test_apertus_sampler.cpp sampler.cpp)
  target_include_directories(test_apertus_sampler PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_apertus_sampler PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_apertus_sampler COMMAND test_apertus_sampler)
endif()
