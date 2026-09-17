# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_llama SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_llama
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_llama SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_llama PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_llama PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
set_target_properties(trtmc_model_llama PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_llama
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS test_llama_chat_template test_llama_native_kv_cache)
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/llama/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_llama
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  set_tests_properties(test_llama_native_kv_cache PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(test_llama_plugin_helpers
    ${PROJECT_SOURCE_DIR}/families/llama/tests/cpp/test_llama_plugin_helpers.cpp
  )
  target_include_directories(test_llama_plugin_helpers PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_llama_plugin_helpers PRIVATE
    trtmc_model_llama
    trtmc_core
  )
  target_compile_options(test_llama_plugin_helpers PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_llama_plugin_helpers COMMAND test_llama_plugin_helpers)

  add_executable(test_llama_pipeline
    ${PROJECT_SOURCE_DIR}/families/llama/tests/cpp/test_llama_pipeline.cpp
  )
  target_include_directories(test_llama_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_llama_pipeline SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_llama_pipeline PRIVATE
    trtmc_model_llama
    trtmc_backend_trt
    trtmc_core
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_llama_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_llama_pipeline COMMAND test_llama_pipeline)
  set_tests_properties(test_llama_pipeline PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  target_sources(trtmc_model_llama PRIVATE
    edge_llm/adapter.cpp
    edge_llm/device_link.cu
  )
  target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_llama PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_llama PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_llama POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_llama>
    VERBATIM
  )
endif()
