# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_qwen3_8_flash_next SHARED plugin.cpp native_task.cpp native_decoder.cpp decoder_layer.cpp experts.cpp positions.cpp ngram.cpp ple_rows.cpp lexical_table.cpp native_visual.cpp tokenizer.cpp sampler.cpp)
target_include_directories(trtmc_model_qwen3_8_flash_next SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_include_directories(trtmc_model_qwen3_8_flash_next PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_model_qwen3_8_flash_next PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
set_target_properties(trtmc_model_qwen3_8_flash_next PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
include("${CMAKE_CURRENT_LIST_DIR}/edge_llm/Adapter.cmake")
install(TARGETS trtmc_model_qwen3_8_flash_next LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_8_flash_next_sdk_cpp ../tests/sdk_consumer.cpp)
  target_link_libraries(test_qwen3_8_flash_next_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_qwen3_8_flash_next_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_qwen3_8_flash_next_lexical_stage ../tests/cpp/lexical_stage.cpp)
  target_link_libraries(test_qwen3_8_flash_next_lexical_stage PRIVATE trtmc_backend_trt)
  target_include_directories(test_qwen3_8_flash_next_lexical_stage SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(test_qwen3_8_flash_next_lexical_stage PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  set_target_properties(test_qwen3_8_flash_next_lexical_stage PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_dependencies(test_qwen3_8_flash_next_sdk_cpp test_qwen3_8_flash_next_lexical_stage)
  set(test_name test_qwen3_8_flash_next_ngram)
  add_executable(${test_name} ../tests/cpp/test_ngram.cpp ngram.cpp)
  target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR})
  add_test(NAME ${test_name} COMMAND ${test_name})
endif()
