# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_m2m_100 SHARED
  decode_runtime.cpp
  kv_cache.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_m2m_100_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_m2m_100 PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_m2m_100 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_m2m_100 PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_m2m_100 PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_m2m_100 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_m2m_100 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/m2m_100"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_m2m_100_request_tokens
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_m2m_100_request_tokens.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
endif()
