# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_omni SHARED
  bpe_tokenizer.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_qwen3_omni
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen3_omni SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_qwen3_omni PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_qwen3_omni PRIVATE
  -Wall -Wextra -Wpedantic
)
set_target_properties(trtmc_model_qwen3_omni PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_omni
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_omni_pipeline
    ${PROJECT_SOURCE_DIR}/families/qwen3_omni/tests/cpp/test_qwen3_omni_pipeline.cpp
  )
  target_include_directories(test_qwen3_omni_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_qwen3_omni_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen3_omni_pipeline PRIVATE
    trtmc_model_qwen3_omni
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_qwen3_omni_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_qwen3_omni_pipeline COMMAND test_qwen3_omni_pipeline)
  set_tests_properties(test_qwen3_omni_pipeline PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()
