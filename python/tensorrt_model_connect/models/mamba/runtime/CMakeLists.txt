# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_mamba SHARED
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  recurrent_state.cpp
  sampler.cpp
  ${TRTMC_MODEL_mamba_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_mamba PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_mamba SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_mamba PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_mamba PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_mamba PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_mamba PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/mamba"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_mamba_recurrent_output_initializers
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_mamba_recurrent_output_initializers.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_mamba_recurrent_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_mamba_recurrent_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_mamba_recurrent_pipeline)
    target_link_libraries(test_mamba_recurrent_pipeline PRIVATE trtmc_model_mamba trtmc_backend_trt)
  endif()
endif()
