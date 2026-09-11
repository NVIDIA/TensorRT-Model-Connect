# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_k2_horizon_uno SHARED
  algorithm.cpp
  bpe_tokenizer.cpp
  chat_template.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_k2_horizon_uno
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_k2_horizon_uno SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_k2_horizon_uno PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_k2_horizon_uno PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_k2_horizon_uno PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_k2_horizon_uno
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS
      test_k2_horizon_uno_algorithm
      test_k2_horizon_uno_chat_template
      test_k2_horizon_uno_runtime_config
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/k2_horizon_uno/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_k2_horizon_uno
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
