# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_lfm2_vl SHARED
  bpe_tokenizer.cpp
  byte_level_decoder.cpp
  chat_templates.cpp
  conv_state.cpp
  conversation.cpp
  hybrid_state.cpp
  kv_cache.cpp
  semantic.cpp
  image_processor.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  pretokenizer.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_lfm2_vl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm2_vl SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lfm2_vl PRIVATE
  trtmc_core
  ICU::uc
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lfm2_vl PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_lfm2_vl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm2_vl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS
      test_lfm2_vl_byte_level_decoder
      test_lfm2_vl_chat_template
      test_lfm2_vl_conversation
      test_lfm2_vl_pretokenizer
      test_lfm2_vl_runtime
      test_lfm2_vl_sampler
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/lfm2_vl/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_lfm2_vl
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  set_tests_properties(test_lfm2_vl_runtime PROPERTIES SKIP_RETURN_CODE 77)
  target_compile_definitions(test_lfm2_vl_conversation PRIVATE
    TRTMC_LFM2_VL_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/families/lfm2_vl/tests/fixtures")
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(lfm2_vl_probe ${PROJECT_SOURCE_DIR}/families/lfm2_vl/tests/cpp/lfm2_vl_probe.cpp)
  target_include_directories(lfm2_vl_probe PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(lfm2_vl_probe SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(lfm2_vl_probe PRIVATE trtmc_model_lfm2_vl trtmc_c nlohmann_json::nlohmann_json)
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(lfm25_vl_logits_probe ${PROJECT_SOURCE_DIR}/families/lfm2_vl/tests/cpp/lfm25_vl_logits_probe.cpp)
  target_include_directories(lfm25_vl_logits_probe PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(lfm25_vl_logits_probe SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(lfm25_vl_logits_probe PRIVATE trtmc_model_lfm2_vl trtmc_backend_trt trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
endif()
