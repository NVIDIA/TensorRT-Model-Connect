# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen SHARED
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
  triattention_kv_cache.cpp
  ${TRTMC_MODEL_qwen_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_qwen PRIVATE
    argmax_kernel.cu
    sparse_multinomial_kernel.cu
    triattention_kernels.cu
  )
endif()
target_include_directories(trtmc_model_qwen PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_qwen SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_qwen PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_qwen PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_qwen PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/qwen"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_c_abi_runtime_regression
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_c_abi_runtime_regression.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_c_abi_runtime_regression)
    target_link_libraries(test_c_abi_runtime_regression PRIVATE trtmc_model_qwen)
  endif()
  trtmc_add_test(test_qwen_tensor_names
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_tensor_names.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_qwen_sampler
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_sampler.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_qwen_sampler)
    target_link_libraries(test_qwen_sampler PRIVATE trtmc_model_qwen)
  endif()
  trtmc_add_test(test_qwen_native_kv_cache
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_native_kv_cache.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
    REQUIRES_GPU
  )
  if(TARGET test_qwen_native_kv_cache)
    target_link_libraries(test_qwen_native_kv_cache PRIVATE trtmc_model_qwen trtmc_backend_trt)
  endif()
endif()
