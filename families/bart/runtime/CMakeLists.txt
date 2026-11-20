# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_bart SHARED
  bpe_tokenizer.cpp
  decode_runtime.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_bart
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_bart SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_bart PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_bart PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_bart PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_bart
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_bart_added_token_whitespace
    ${PROJECT_SOURCE_DIR}/families/bart/tests/cpp/test_bart_added_token_whitespace.cpp
    bpe_tokenizer.cpp
  )
  target_include_directories(test_bart_added_token_whitespace PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_bart_added_token_whitespace PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_bart_added_token_whitespace PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME bart_added_token_whitespace COMMAND test_bart_added_token_whitespace)

  add_executable(test_bart_runtime_config
    ${PROJECT_SOURCE_DIR}/families/bart/tests/cpp/test_bart_runtime_config.cpp
  )
  target_include_directories(test_bart_runtime_config PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_bart_runtime_config PRIVATE trtmc_model_bart)
  target_compile_options(test_bart_runtime_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME bart_runtime_config COMMAND test_bart_runtime_config)

  # CPU CUDA stubs live in the test, so allocation failures can be injected
  # without a GPU or cudart.
  add_executable(test_bart_cross_kv_alloc
    ${PROJECT_SOURCE_DIR}/families/bart/tests/cpp/test_bart_cross_kv_alloc.cpp
  )
  target_include_directories(test_bart_cross_kv_alloc PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_bart_cross_kv_alloc SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_compile_options(test_bart_cross_kv_alloc PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME bart_cross_kv_alloc COMMAND test_bart_cross_kv_alloc)

  add_executable(test_bart_task_contract
    ${PROJECT_SOURCE_DIR}/families/bart/tests/cpp/test_task_contract.cpp
    pipeline.cpp
    decode_runtime.cpp
  )
  target_include_directories(test_bart_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_bart_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_bart_task_contract PRIVATE
    trtmc_core
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_bart_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME bart_task_contract COMMAND test_bart_task_contract)
  set_tests_properties(bart_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_bart_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/bart/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_bart_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_bart_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_bart_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()

  add_dependencies(test_bart_task_contract
    test_bart_sdk_c test_bart_sdk_cpp
  )
endif()
