# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gpt2 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_gpt2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_gpt2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_gpt2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_gpt2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_gpt2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_gpt2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_gpt2_task_config
    ${PROJECT_SOURCE_DIR}/families/gpt2/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_gpt2_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_gpt2_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME gpt2_task_config COMMAND test_gpt2_task_config)
  set_tests_properties(gpt2_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_gpt2_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/gpt2/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_gpt2_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_gpt2_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_gpt2_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_gpt2_task_config test_gpt2_sdk_c test_gpt2_sdk_cpp)

  add_executable(test_gpt2_parallel_kv_contract
    ${PROJECT_SOURCE_DIR}/families/gpt2/tests/cpp/test_parallel_kv_contract.cpp
  )
  target_include_directories(test_gpt2_parallel_kv_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_gpt2_parallel_kv_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_include_directories(test_gpt2_parallel_kv_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_gpt2_parallel_kv_contract PRIVATE trtmc_model_gpt2)
  target_compile_options(test_gpt2_parallel_kv_contract PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME gpt2_parallel_kv_contract COMMAND test_gpt2_parallel_kv_contract)
endif()
