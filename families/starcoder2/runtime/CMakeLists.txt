# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_starcoder2 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_starcoder2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_starcoder2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_starcoder2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_starcoder2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_starcoder2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_starcoder2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  set(test_name test_starcoder2_tokenizer_contract)
  add_executable(${test_name}
    ${PROJECT_SOURCE_DIR}/families/starcoder2/tests/cpp/${test_name}.cpp
  )
  target_include_directories(${test_name} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(${test_name} PRIVATE
    trtmc_model_starcoder2
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
endif()

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_starcoder2 SHARED cli.cpp)
target_include_directories(trtmc_cli_starcoder2 PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_starcoder2 PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_starcoder2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_starcoder2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_starcoder2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_starcoder2_cli
    ${PROJECT_SOURCE_DIR}/families/starcoder2/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_starcoder2_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_starcoder2_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_starcoder2_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME starcoder2_cli COMMAND test_starcoder2_cli)
  set_tests_properties(starcoder2_cli PROPERTIES LABELS cpu)
endif()
