# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_deepseek_v2 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_deepseek_v2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_deepseek_v2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_deepseek_v2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_deepseek_v2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_deepseek_v2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_deepseek_v2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_deepseek_v2_parallel_kv_contract
    ${PROJECT_SOURCE_DIR}/families/deepseek_v2/tests/cpp/test_parallel_kv_contract.cpp
  )
  target_include_directories(test_deepseek_v2_parallel_kv_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_deepseek_v2_parallel_kv_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_deepseek_v2_parallel_kv_contract PRIVATE
    trtmc_model_deepseek_v2
  )
  target_compile_options(test_deepseek_v2_parallel_kv_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME deepseek_v2_parallel_kv_contract
    COMMAND test_deepseek_v2_parallel_kv_contract
  )
endif()
