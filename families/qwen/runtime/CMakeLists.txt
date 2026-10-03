# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  embedding_factory.cpp
  embedding_pipeline.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_qwen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_qwen PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_qwen PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_qwen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS
      test_qwen_embedding_pipeline
      test_qwen_native_kv_cache
      test_qwen_sampler
      test_qwen_tensor_names
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/qwen/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_qwen
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  set_tests_properties(test_qwen_native_kv_cache PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(qwen_logits_trace
    ${PROJECT_SOURCE_DIR}/families/qwen/tests/cpp/logits_trace.cpp
  )
  target_include_directories(qwen_logits_trace PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(qwen_logits_trace SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(qwen_logits_trace PRIVATE
    trtmc_runtime
    trtmc_model_qwen
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(qwen_logits_trace PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen_task_config
    ${PROJECT_SOURCE_DIR}/families/qwen/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_qwen_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_qwen_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME qwen_task_config COMMAND test_qwen_task_config)
  set_tests_properties(qwen_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_qwen_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/qwen/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_qwen_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_qwen_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_qwen_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_qwen_task_config test_qwen_sdk_c test_qwen_sdk_cpp)

  add_executable(qwen_embedding_consumer
    ${PROJECT_SOURCE_DIR}/families/qwen/tests/cpp/qwen_embedding_consumer.cpp)
  target_link_libraries(qwen_embedding_consumer PRIVATE trtmc_c)
endif()
