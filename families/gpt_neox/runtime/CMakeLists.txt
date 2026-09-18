# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gpt_neox SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_gpt_neox
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_gpt_neox SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_gpt_neox PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_gpt_neox PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_gpt_neox PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_gpt_neox
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  set(test_name test_gpt_neox_attention_mask)
  add_executable(${test_name}
    ${PROJECT_SOURCE_DIR}/families/gpt_neox/tests/cpp/${test_name}.cpp
  )
  target_include_directories(${test_name} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(${test_name} PRIVATE
    trtmc_model_gpt_neox
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_gpt_neox_task_config
    ${PROJECT_SOURCE_DIR}/families/gpt_neox/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_gpt_neox_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_gpt_neox_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME gpt_neox_task_config COMMAND test_gpt_neox_task_config)
  set_tests_properties(gpt_neox_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_gpt_neox_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/gpt_neox/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_gpt_neox_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_gpt_neox_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_gpt_neox_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_gpt_neox_task_config test_gpt_neox_sdk_c test_gpt_neox_sdk_cpp)

endif()
