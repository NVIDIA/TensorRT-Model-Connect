# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_xglm SHARED
  unigram_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_xglm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_xglm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_xglm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_xglm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_xglm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_xglm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_xglm_task_config
    ${PROJECT_SOURCE_DIR}/families/xglm/tests/cpp/test_task_config.cpp
  )
  target_include_directories(test_xglm_task_config PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_xglm_task_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME xglm_task_config COMMAND test_xglm_task_config)
  set_tests_properties(xglm_task_config PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_xglm_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/xglm/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_xglm_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_xglm_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_xglm_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_xglm_task_config test_xglm_sdk_c test_xglm_sdk_cpp)

endif()
