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

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_codegen SHARED cli.cpp)
target_include_directories(trtmc_cli_codegen PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_codegen PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_codegen PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_codegen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_codegen LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_codegen_cli
    ${PROJECT_SOURCE_DIR}/families/codegen/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_codegen_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_codegen_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_codegen_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME codegen_cli COMMAND test_codegen_cli)
  set_tests_properties(codegen_cli PROPERTIES LABELS cpu)
endif()
