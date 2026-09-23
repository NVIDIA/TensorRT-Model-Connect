# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_fnet SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_fnet
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_fnet SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_fnet PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_fnet PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_fnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_fnet
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# The owner command adapter loads the model through the existing runtime boundary.
add_library(trtmc_cli_fnet SHARED cli.cpp)
target_include_directories(trtmc_cli_fnet PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_fnet PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_fnet PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_fnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_fnet LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_fnet_cli
    ${PROJECT_SOURCE_DIR}/families/fnet/tests/cpp/test_cli.cpp
    cli.cpp
  )
  target_include_directories(test_fnet_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_fnet_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_fnet_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME fnet_cli COMMAND test_fnet_cli)
  set_tests_properties(fnet_cli PROPERTIES LABELS cpu)
endif()
