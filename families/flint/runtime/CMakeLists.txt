# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_flint SHARED plugin.cpp pipeline.cpp bpe_tokenizer.cpp sampler.cpp)
target_include_directories(trtmc_model_flint PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_flint SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_flint PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS} ICU::uc ICU::i18n)
target_compile_options(trtmc_model_flint PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_flint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_flint LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(flint_tokenizer_probe ../tests/cpp/tokenizer_probe.cpp)
  target_include_directories(flint_tokenizer_probe PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(flint_tokenizer_probe PRIVATE trtmc_model_flint nlohmann_json::nlohmann_json)
  set_target_properties(flint_tokenizer_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME flint_tokenizer COMMAND ${Python3_EXECUTABLE} -m pytest
    ${PROJECT_SOURCE_DIR}/families/flint/tests/test_tokenizer.py -q)
  set_tests_properties(flint_tokenizer PROPERTIES ENVIRONMENT
    "TRTMC_FLINT_TOKENIZER_PROBE=$<TARGET_FILE:flint_tokenizer_probe>;PYTHONPATH=${PROJECT_SOURCE_DIR}/core/builder:${PROJECT_SOURCE_DIR}/apps/benchmark:${PROJECT_SOURCE_DIR}")
  add_executable(flint_logits_trace ../tests/cpp/logits_trace.cpp)
  target_include_directories(flint_logits_trace PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(flint_logits_trace PRIVATE
    trtmc_runtime trtmc_model_flint nlohmann_json::nlohmann_json)
  set_target_properties(flint_logits_trace PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set_target_properties(flint_logits_trace PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
  install(TARGETS flint_logits_trace RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
