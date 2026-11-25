# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_h SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  distributed_runtime.cpp
  hybrid_state.cpp
  kv_cache.cpp
  paired.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  recurrent_state.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_nemotron_h
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_nemotron_h SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_nemotron_h PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_nemotron_h PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_nemotron_h PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron_h
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  set(test_name test_nemotron_h_recurrent_output_initializers)
  add_executable(${test_name}
    ${PROJECT_SOURCE_DIR}/families/nemotron_h/tests/cpp/${test_name}.cpp
  )
  target_include_directories(${test_name} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(${test_name} PRIVATE
    trtmc_model_nemotron_h
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME ${test_name} COMMAND ${test_name})
endif()

include(edge_llm/Adapter.cmake)

if(TRTMC_BUILD_TESTS)
  # CPU-backed KV-cache test: CUDA calls are served from host memory, and an
  # overlapping copy fails the test.
  add_executable(test_nemotron_h_kv_cache
    ${PROJECT_SOURCE_DIR}/families/nemotron_h/tests/cpp/test_kv_cache.cpp
    kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
  )
  target_include_directories(test_nemotron_h_kv_cache PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_nemotron_h_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_nemotron_h_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME nemotron_h_kv_cache COMMAND test_nemotron_h_kv_cache)
endif()
