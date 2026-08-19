# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_h SHARED
  chat_templates.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  recurrent_state.cpp
  sampler.cpp
  ${TRTMC_MODEL_nemotron_h_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_nemotron_h PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_nemotron_h SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_nemotron_h PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_nemotron_h PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_nemotron_h PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_nemotron_h PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/nemotron_h"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_nemotron_h_backend_cuda_graph
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_h_backend_cuda_graph.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
    REQUIRES_GPU
  )
  if(TARGET test_nemotron_h_backend_cuda_graph)
    target_link_libraries(test_nemotron_h_backend_cuda_graph PRIVATE trtmc_backend_trt)
  endif()
  trtmc_add_test(test_nemotron_h_recurrent_output_initializers
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_h_recurrent_output_initializers.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_nemotron_h_recurrent_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_h_recurrent_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_nemotron_h_recurrent_pipeline)
    target_link_libraries(test_nemotron_h_recurrent_pipeline PRIVATE trtmc_model_nemotron_h trtmc_backend_trt)
  endif()
endif()
