# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gemma SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_gemma
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_gemma SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_gemma PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_gemma PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
set_target_properties(trtmc_model_gemma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_gemma
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS test_gemma_pipeline test_gemma_sampler)
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/gemma/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_gemma
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  set_tests_properties(test_gemma_pipeline PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Optional complete-network offload; all model-specific orchestration stays here.
if(TARGET EdgeLLM::Core)
  target_sources(trtmc_model_gemma PRIVATE edge_llm/adapter.cpp edge_llm/device_link.cu)
  target_compile_definitions(trtmc_model_gemma PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_gemma PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_gemma PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_gemma POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_gemma> VERBATIM)
  if(TRTMC_BUILD_TESTS)
    target_compile_definitions(test_gemma_sampler PRIVATE TRTMC_HAS_EDGE_LLM=1)
    target_link_libraries(test_gemma_sampler PRIVATE EdgeLLM::Core)
  endif()
endif()
