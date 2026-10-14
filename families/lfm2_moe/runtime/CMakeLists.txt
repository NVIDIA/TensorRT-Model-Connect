# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_lfm2_moe SHARED
  bpe_tokenizer.cpp
  byte_level_decoder.cpp
  chat_templates.cpp
  conv_state.cpp
  conversation.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  pretokenizer.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_lfm2_moe
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm2_moe SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lfm2_moe PRIVATE
  trtmc_core
  ICU::uc
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lfm2_moe PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_lfm2_moe PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm2_moe
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  if(TARGET trtmc_backend_trt)
    add_executable(lfm2_moe_logits_probe
      ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/cpp/lfm25_logits_probe.cpp)
    target_include_directories(lfm2_moe_logits_probe PRIVATE
      ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(lfm2_moe_logits_probe PRIVATE
      trtmc_model_lfm2_moe trtmc_core trtmc_backend_trt nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  endif()

  foreach(test_name IN ITEMS
      test_lfm2_moe_byte_level_decoder
      test_lfm2_moe_chat_template
      test_lfm2_moe_pretokenizer
      test_lfm2_moe_runtime
      test_lfm2_moe_sampler
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_lfm2_moe
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_executable(test_lfm25_moe_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/cpp/test_lfm25_moe_bpe_flags.cpp)
  target_include_directories(test_lfm25_moe_bpe_flags PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_lfm25_moe_bpe_flags PRIVATE trtmc_model_lfm2_moe nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_moe_bpe_flags COMMAND test_lfm25_moe_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/fixtures/lfm25-bpe-flags.json)
  add_executable(test_lfm25_moe_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/cpp/test_lfm25_moe_tokenizer.cpp)
  target_include_directories(test_lfm25_moe_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_lfm25_moe_tokenizer PRIVATE trtmc_model_lfm2_moe nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_moe_tokenizer COMMAND test_lfm25_moe_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/fixtures/lfm25-tokenizer.json)
  add_test(NAME test_lfm25_moe_tokenizer_ids COMMAND test_lfm25_moe_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2_moe/tests/fixtures/lfm25-tokenizer.json --token-ids)
  set_tests_properties(test_lfm25_moe_tokenizer_ids PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(test_lfm2_moe_runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
