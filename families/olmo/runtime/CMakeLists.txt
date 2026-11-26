# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_olmo SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_olmo
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_olmo SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_olmo PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_olmo PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_olmo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_olmo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_olmo_prompt_tokenization
    ${PROJECT_SOURCE_DIR}/families/olmo/tests/cpp/test_olmo_prompt_tokenization.cpp)
  target_include_directories(test_olmo_prompt_tokenization PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_olmo_prompt_tokenization PRIVATE trtmc_model_olmo nlohmann_json::nlohmann_json)
  add_test(NAME olmo_prompt_tokenization COMMAND test_olmo_prompt_tokenization)
  set_tests_properties(olmo_prompt_tokenization PROPERTIES LABELS "cpu")
  add_executable(test_olmo_task_config
    ${PROJECT_SOURCE_DIR}/families/olmo/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_olmo_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_olmo_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME olmo_task_config COMMAND test_olmo_task_config)
  set_tests_properties(olmo_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_olmo_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/olmo/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_olmo_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_olmo_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_olmo_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_olmo_task_config test_olmo_sdk_c test_olmo_sdk_cpp)

endif()

if(TRTMC_BUILD_TESTS)
  # CPU-backed KV-cache test: CUDA calls are served from host memory, and an
  # overlapping copy fails the test.
  add_executable(test_olmo_kv_cache
    ${PROJECT_SOURCE_DIR}/families/olmo/tests/cpp/test_kv_cache.cpp
    kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
  )
  target_include_directories(test_olmo_kv_cache PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_olmo_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_olmo_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME olmo_kv_cache COMMAND test_olmo_kv_cache)
endif()
