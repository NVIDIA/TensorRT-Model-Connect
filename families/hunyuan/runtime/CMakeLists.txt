# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hunyuan SHARED
  bpe_tokenizer.cpp
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

if(TRTMC_BUILD_TESTS AND TARGET EdgeLLM::Core)
  add_executable(test_hunyuan_contract
    ${PROJECT_SOURCE_DIR}/families/hunyuan/tests/cpp/test_hunyuan_contract.cpp)
  target_include_directories(test_hunyuan_contract PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
    $<TARGET_PROPERTY:EdgeLLM::Core,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(test_hunyuan_contract PRIVATE
    trtmc_model_hunyuan trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  add_test(NAME test_hunyuan_contract COMMAND test_hunyuan_contract)
endif()
