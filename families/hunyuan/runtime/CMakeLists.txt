# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_hunyuan SHARED
  bpe_tokenizer.cpp
  mt2_pre_tokenizer.cpp
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_hunyuan
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_hunyuan SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_hunyuan PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ICU::uc
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_hunyuan PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
set_target_properties(trtmc_model_hunyuan PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_hunyuan
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

include(edge_llm/Adapter.cmake)

if(TRTMC_BUILD_TESTS)
  add_executable(test_hunyuan_contract
    ${PROJECT_SOURCE_DIR}/families/hunyuan/tests/cpp/test_hunyuan_contract.cpp)
  target_include_directories(test_hunyuan_contract PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  if(DEFINED _hunyuan_sdk)
    if(TARGET TensorRTEdgeLLM::Core)
      target_compile_definitions(test_hunyuan_contract PRIVATE TRTMC_HUNYUAN_EDGE_012=1)
    endif()
    target_compile_definitions(test_hunyuan_contract PRIVATE TRTMC_HAS_EDGE_LLM=1)
    target_include_directories(test_hunyuan_contract PRIVATE
      $<TARGET_PROPERTY:${_hunyuan_sdk}::Core,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  target_link_libraries(test_hunyuan_contract PRIVATE
    trtmc_model_hunyuan trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  add_test(NAME test_hunyuan_contract COMMAND test_hunyuan_contract)
endif()

if(TRTMC_BUILD_TESTS)
  # CPU-backed KV-cache test: CUDA calls are served from host memory, and an
  # overlapping copy fails the test.
  add_executable(test_hunyuan_kv_cache
    ${PROJECT_SOURCE_DIR}/families/hunyuan/tests/cpp/test_kv_cache.cpp
    kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
  )
  target_include_directories(test_hunyuan_kv_cache PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_hunyuan_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_hunyuan_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hunyuan_kv_cache COMMAND test_hunyuan_kv_cache)
endif()
