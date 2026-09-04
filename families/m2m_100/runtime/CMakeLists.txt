# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_m2m_100 SHARED
  decode_runtime.cpp
  kv_cache.cpp
  plugin.cpp
  plugin_helpers.cpp
  unigram_tokenizer.cpp
)

target_include_directories(trtmc_model_m2m_100
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_m2m_100 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_m2m_100 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_m2m_100 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_m2m_100 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_m2m_100
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_m2m_100_runtime_config
    ${PROJECT_SOURCE_DIR}/families/m2m_100/tests/cpp/test_m2m_100_runtime_config.cpp
  )
  target_include_directories(test_m2m_100_runtime_config PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_m2m_100_runtime_config PRIVATE trtmc_model_m2m_100)
  target_compile_options(test_m2m_100_runtime_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME m2m_100_runtime_config COMMAND test_m2m_100_runtime_config)

  add_executable(test_m2m_100_tokenizer_schema
    ${PROJECT_SOURCE_DIR}/families/m2m_100/tests/cpp/test_m2m_100_tokenizer_schema.cpp
  )
  target_include_directories(test_m2m_100_tokenizer_schema PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_m2m_100_tokenizer_schema PRIVATE trtmc_model_m2m_100)
  target_compile_options(test_m2m_100_tokenizer_schema PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME m2m_100_tokenizer_schema COMMAND test_m2m_100_tokenizer_schema)
endif()
