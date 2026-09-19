# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_stablelm SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_stablelm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_stablelm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_stablelm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_stablelm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_stablelm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_stablelm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_stablelm SHARED cli.cpp)
target_include_directories(trtmc_cli_stablelm PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_stablelm PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_stablelm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_stablelm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_stablelm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_stablelm_cli
    ${PROJECT_SOURCE_DIR}/families/stablelm/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_stablelm_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_stablelm_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_stablelm_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stablelm_cli COMMAND test_stablelm_cli)
  set_tests_properties(stablelm_cli PROPERTIES LABELS cpu)
endif()
