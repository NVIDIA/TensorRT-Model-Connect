# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_lfm2 SHARED
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

target_include_directories(trtmc_model_lfm2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lfm2 PRIVATE
  trtmc_core
  ICU::uc
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lfm2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_lfm2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  if(TARGET trtmc_backend_trt)
    add_executable(lfm2_logits_probe
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/lfm25_logits_probe.cpp)
    target_include_directories(lfm2_logits_probe PRIVATE
      ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(lfm2_logits_probe PRIVATE
      trtmc_model_lfm2 trtmc_core trtmc_backend_trt nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  endif()

  foreach(test_name IN ITEMS
      test_lfm2_byte_level_decoder
      test_lfm2_chat_template
      test_lfm2_pretokenizer
      test_lfm2_runtime
      test_lfm2_sampler
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_lfm2
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_executable(test_lfm25_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm25_bpe_flags.cpp)
  target_include_directories(test_lfm25_bpe_flags PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_lfm25_bpe_flags PRIVATE trtmc_model_lfm2 nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_bpe_flags COMMAND test_lfm25_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-bpe-flags.json)
  add_executable(test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm25_tokenizer.cpp)
  target_include_directories(test_lfm25_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_lfm25_tokenizer PRIVATE trtmc_model_lfm2 nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_tokenizer COMMAND test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-tokenizer.json)
  add_custom_target(test_lfm25_tokenizer_ids DEPENDS test_lfm25_tokenizer)
  add_test(NAME test_lfm25_tokenizer_ids COMMAND test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-tokenizer.json --token-ids)
  set_tests_properties(test_lfm25_tokenizer_ids PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(test_lfm2_runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
