# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_h3 SHARED
  config_schema.cpp
  pipeline.cpp
  plugin.cpp
  ${TRTMC_MODEL_minimax_h3_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_minimax_h3 PRIVATE
    torch_cuda_normal.cu
  )
endif()
target_include_directories(trtmc_model_minimax_h3 PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_minimax_h3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_minimax_h3 PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_minimax_h3 PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_minimax_h3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_minimax_h3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/minimax_h3"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_minimax_h3_config_schema
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_minimax_h3_config_schema.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_minimax_h3_config_schema)
    target_link_libraries(test_minimax_h3_config_schema PRIVATE trtmc_model_minimax_h3)
  endif()
  trtmc_add_test(test_minimax_h3_math
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_minimax_h3_math.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_minimax_h3_math)
    target_link_libraries(test_minimax_h3_math PRIVATE trtmc_model_minimax_h3)
  endif()
  trtmc_add_test(test_minimax_h3_cuda_rng
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_minimax_h3_cuda_rng.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_minimax_h3_cuda_rng)
    target_link_libraries(test_minimax_h3_cuda_rng PRIVATE trtmc_model_minimax_h3)
  endif()
  trtmc_add_test(test_minimax_h3_trt_module_bind_external
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_minimax_h3_trt_module_bind_external.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
    REQUIRES_GPU
  )
  if(TARGET test_minimax_h3_trt_module_bind_external)
    target_link_libraries(test_minimax_h3_trt_module_bind_external PRIVATE trtmc_backend_trt)
  endif()
endif()
