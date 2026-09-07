# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_smollm3 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_smollm3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_smollm3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_smollm3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_smollm3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_smollm3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_smollm3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS test_smollm3_chat_template test_smollm3_native_kv_cache)
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/smollm3/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_smollm3
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  set_tests_properties(test_smollm3_native_kv_cache PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(test_smollm3_plugin_helpers
    ${PROJECT_SOURCE_DIR}/families/smollm3/tests/cpp/test_smollm3_plugin_helpers.cpp
  )
  target_include_directories(test_smollm3_plugin_helpers PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_smollm3_plugin_helpers PRIVATE
    trtmc_model_smollm3
    trtmc_core
  )
  target_compile_options(test_smollm3_plugin_helpers PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_smollm3_plugin_helpers COMMAND test_smollm3_plugin_helpers)

  add_executable(test_smollm3_pipeline
    ${PROJECT_SOURCE_DIR}/families/smollm3/tests/cpp/test_smollm3_pipeline.cpp
  )
  target_include_directories(test_smollm3_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_smollm3_pipeline SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_smollm3_pipeline PRIVATE
    trtmc_model_smollm3
    trtmc_backend_trt
    trtmc_core
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_smollm3_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_smollm3_pipeline COMMAND test_smollm3_pipeline)
  set_tests_properties(test_smollm3_pipeline PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()
