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

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_xglm SHARED cli.cpp)
target_include_directories(trtmc_cli_xglm PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_xglm PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_xglm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_xglm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_xglm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_xglm_cli
    ${PROJECT_SOURCE_DIR}/families/xglm/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_xglm_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_xglm_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_xglm_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME xglm_cli COMMAND test_xglm_cli)
  set_tests_properties(xglm_cli PROPERTIES LABELS cpu)
endif()
