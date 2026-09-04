# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_8 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  recurrent_state.cpp
  sampler.cpp
)
target_include_directories(trtmc_model_qwen3_8 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_link_libraries(trtmc_model_qwen3_8 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_qwen3_8 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_qwen3_8 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_8 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_8_recurrent_output_initializers
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_recurrent_output_initializers.cpp
  )
  target_include_directories(test_qwen3_8_recurrent_output_initializers PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_qwen3_8_recurrent_output_initializers PRIVATE
    trtmc_model_qwen3_8
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME test_qwen3_8_recurrent_output_initializers
    COMMAND test_qwen3_8_recurrent_output_initializers
  )

  add_executable(test_qwen3_8_recurrent_pipeline
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_recurrent_pipeline.cpp
  )
  target_include_directories(test_qwen3_8_recurrent_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_qwen3_8_recurrent_pipeline SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen3_8_recurrent_pipeline PRIVATE
    trtmc_model_qwen3_8
    trtmc_backend_trt
    trtmc_core
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME test_qwen3_8_recurrent_pipeline COMMAND test_qwen3_8_recurrent_pipeline)
  set_tests_properties(test_qwen3_8_recurrent_pipeline PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()
