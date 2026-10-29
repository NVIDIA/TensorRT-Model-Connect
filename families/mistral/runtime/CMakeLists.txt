# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_mistral SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_mistral
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_mistral SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_mistral PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_mistral PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_mistral PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_mistral
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_mistral_pipeline
    ${PROJECT_SOURCE_DIR}/families/mistral/tests/cpp/test_mistral_pipeline.cpp
  )
  target_include_directories(test_mistral_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_mistral_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_mistral_pipeline PRIVATE
    trtmc_model_mistral trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_mistral_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_mistral_pipeline COMMAND test_mistral_pipeline)
  set_tests_properties(test_mistral_pipeline PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
endif()
