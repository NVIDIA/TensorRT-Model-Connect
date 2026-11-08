# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lfm3 SHARED
  bpe_tokenizer.cpp
  byte_level_decoder.cpp
  chat_templates.cpp
  conv_state.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  pretokenizer.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_lfm3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lfm3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lfm3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_lfm3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS
      test_lfm3_byte_level_decoder
      test_lfm3_chat_template
      test_lfm3_pretokenizer
      test_lfm3_tokenizer_checkpoint
      test_lfm3_runtime
      test_lfm3_sampler
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/lfm3/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_lfm3
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    # Note (Jiaxin Deng): E2E runs the checkpoint tokenizer after downloading its pinned assets.
    if(NOT test_name STREQUAL "test_lfm3_tokenizer_checkpoint")
      add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
  endforeach()
  set_tests_properties(test_lfm3_runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
