# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_llama SHARED
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
  triattention_kv_cache.cpp
  ${TRTMC_MODEL_llama_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_llama PRIVATE
    argmax_kernel.cu
    sparse_multinomial_kernel.cu
    triattention_kernels.cu
  )
endif()
target_include_directories(trtmc_model_llama PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_llama SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_llama PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_llama PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_llama PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/llama"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_llama_plugin_helpers
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_llama_plugin_helpers.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_llama_plugin_helpers)
    target_sources(test_llama_plugin_helpers PRIVATE
      plugin_helpers.cpp
    )
  endif()
  trtmc_add_test(test_llama_chat_template
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_llama_chat_template.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_llama_chat_template)
    target_link_libraries(test_llama_chat_template PRIVATE trtmc_model_llama)
  endif()
  trtmc_add_test(test_llama_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_llama_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_llama_pipeline)
    target_link_libraries(test_llama_pipeline PRIVATE trtmc_model_llama trtmc_backend_trt)
  endif()
  trtmc_add_test(test_llama_native_kv_cache
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_llama_native_kv_cache.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_llama_native_kv_cache)
    target_link_libraries(test_llama_native_kv_cache PRIVATE trtmc_model_llama)
  endif()
endif()
