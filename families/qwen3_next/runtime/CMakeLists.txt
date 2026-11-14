# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_qwen3_next SHARED plugin.cpp pipeline.cpp bpe_tokenizer.cpp sampler.cpp routing.cpp)
target_include_directories(trtmc_model_qwen3_next PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_qwen3_next SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_qwen3_next PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n
  ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
target_compile_options(trtmc_model_qwen3_next PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_qwen3_next PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_qwen3_next LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(test_qwen3_next_sampler ../tests/cpp/test_sampler.cpp)
  target_include_directories(test_qwen3_next_sampler PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_qwen3_next_sampler PRIVATE trtmc_model_qwen3_next)
  set_target_properties(test_qwen3_next_sampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME qwen3_next_sampler COMMAND test_qwen3_next_sampler)

  add_executable(qwen3_next_pipeline_probe ../tests/cpp/pipeline_probe.cpp)
  target_include_directories(qwen3_next_pipeline_probe PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(qwen3_next_pipeline_probe SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(qwen3_next_pipeline_probe PRIVATE trtmc_model_qwen3_next trtmc_core nlohmann_json::nlohmann_json)
  set_target_properties(qwen3_next_pipeline_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME qwen3_next_pipeline_retry COMMAND ${Python3_EXECUTABLE} -m pytest
    ${PROJECT_SOURCE_DIR}/families/qwen3_next/tests/test_runtime.py -q)
  add_test(NAME qwen3_next_tokenizer COMMAND ${Python3_EXECUTABLE} -m pytest
    ${PROJECT_SOURCE_DIR}/families/qwen3_next/tests/test_tokenizer.py -q)
  set_tests_properties(qwen3_next_pipeline_retry qwen3_next_tokenizer PROPERTIES ENVIRONMENT
    "TRTMC_RUNTIME_ROOT=$<TARGET_FILE_DIR:qwen3_next_pipeline_probe>;PYTHONPATH=${PROJECT_SOURCE_DIR}/core/builder:${PROJECT_SOURCE_DIR}/apps/benchmark:${PROJECT_SOURCE_DIR}")
  add_executable(qwen3_next_logits_trace ../tests/cpp/logits_trace.cpp)
  target_include_directories(qwen3_next_logits_trace PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(qwen3_next_logits_trace PRIVATE
    trtmc_runtime trtmc_model_qwen3_next nlohmann_json::nlohmann_json)
  set_target_properties(qwen3_next_logits_trace PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(qwen3_next_tokenizer_probe ../tests/cpp/tokenizer_probe.cpp)
  target_include_directories(qwen3_next_tokenizer_probe PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(qwen3_next_tokenizer_probe PRIVATE trtmc_model_qwen3_next nlohmann_json::nlohmann_json)
  set_target_properties(qwen3_next_tokenizer_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(qwen3_next_routing_probe ../tests/cpp/routing_probe.cpp)
  target_include_directories(qwen3_next_routing_probe PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(qwen3_next_routing_probe PRIVATE trtmc_model_qwen3_next nlohmann_json::nlohmann_json)
  set_target_properties(qwen3_next_routing_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
