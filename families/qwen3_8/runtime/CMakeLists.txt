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
target_compile_options(trtmc_model_qwen3_8 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
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

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  target_sources(trtmc_model_qwen3_8 PRIVATE
    edge_llm/adapter.cpp
    edge_llm/device_link.cu
  )
  target_compile_definitions(trtmc_model_qwen3_8 PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_qwen3_8 PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_qwen3_8 PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_8_edge_contract
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_edge_contract.cpp
  )
  target_include_directories(test_qwen3_8_edge_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  add_test(NAME test_qwen3_8_edge_contract COMMAND test_qwen3_8_edge_contract)
endif()

if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_qwen3_8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_8>
    VERBATIM
  )
  if(TRTMC_BUILD_TESTS)
    add_executable(test_qwen3_8_edge_request
      ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_edge_request.cpp
    )
    target_include_directories(test_qwen3_8_edge_request PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(test_qwen3_8_edge_request PRIVATE EdgeLLM::Core)
    add_test(NAME test_qwen3_8_edge_request COMMAND test_qwen3_8_edge_request)
  endif()
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_8_edge_factory
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_edge_factory.cpp
  )
  target_include_directories(test_qwen3_8_edge_factory PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen3_8_edge_factory PRIVATE
    trtmc_model_qwen3_8 trtmc_core nlohmann_json::nlohmann_json
  )
  if(TARGET EdgeLLM::Core)
    target_compile_definitions(test_qwen3_8_edge_factory PRIVATE TRTMC_HAS_EDGE_LLM=1)
  endif()
  add_test(NAME test_qwen3_8_edge_factory COMMAND test_qwen3_8_edge_factory)
endif()

# Explicit GPU integration driver, not an unprovisioned default CTest.
if(TRTMC_BUILD_TESTS)
  add_executable(qwen3_8_edge_inference
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_edge_inference.cpp
  )
  target_link_libraries(qwen3_8_edge_inference PRIVATE
    trtmc_runtime trtmc_core nlohmann_json::nlohmann_json
  )
endif()
