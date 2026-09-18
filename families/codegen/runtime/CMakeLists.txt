# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_codegen SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_codegen SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_codegen PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_codegen PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_codegen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_codegen
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_codegen_task_config
    ${PROJECT_SOURCE_DIR}/families/codegen/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_codegen_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_codegen_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME codegen_task_config COMMAND test_codegen_task_config)
  set_tests_properties(codegen_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_codegen_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/codegen/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_codegen_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_codegen_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_codegen_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_codegen_task_config test_codegen_sdk_c test_codegen_sdk_cpp)

endif()
