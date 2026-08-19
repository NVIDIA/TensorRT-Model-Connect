# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gemma SHARED
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
  triattention_kv_cache.cpp
  ${TRTMC_MODEL_gemma_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_gemma PRIVATE
    argmax_kernel.cu
    sparse_multinomial_kernel.cu
    triattention_kernels.cu
  )
endif()
target_include_directories(trtmc_model_gemma PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_gemma SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_gemma PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_gemma PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_gemma PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_gemma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/gemma"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_gemma_sampler
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_gemma_sampler.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_gemma_sampler)
    target_link_libraries(test_gemma_sampler PRIVATE trtmc_model_gemma)
  endif()
  trtmc_add_test(test_gemma_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_gemma_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_gemma_pipeline)
    target_link_libraries(test_gemma_pipeline PRIVATE trtmc_model_gemma)
  endif()
endif()
