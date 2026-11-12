# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_glm SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_glm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_glm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_glm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_glm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_glm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_glm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_glm_pipeline
    ${PROJECT_SOURCE_DIR}/families/glm/tests/cpp/test_glm_pipeline.cpp
  )
  target_include_directories(test_glm_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_glm_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_glm_pipeline PRIVATE
    trtmc_model_glm trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_glm_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_glm_pipeline COMMAND test_glm_pipeline)
  set_tests_properties(test_glm_pipeline PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
endif()
