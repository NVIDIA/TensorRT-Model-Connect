# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wan_t2v SHARED
  diffusion_helpers.cpp
  gpu_matmul.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_wan_t2v_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_wan_t2v PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_wan_t2v SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
if(NOT TRTMC_CUBLAS_LIBRARY)
  message(FATAL_ERROR "Model 'wan_t2v' requires cuBLAS, but it was not found")
endif()
target_link_libraries(trtmc_model_wan_t2v PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${TRTMC_CUBLAS_LIBRARY}
)
target_compile_definitions(trtmc_model_wan_t2v PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_wan_t2v PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_wan_t2v PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/wan_t2v"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
set_source_files_properties(pipeline.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_wan_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_wan_pipeline)
    target_link_libraries(test_wan_pipeline PRIVATE trtmc_model_wan_t2v)
  endif()
  trtmc_add_test(test_wan_c_abi_runtime_regression
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_c_abi_runtime_regression.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_wan_c_abi_runtime_regression)
    target_link_libraries(test_wan_c_abi_runtime_regression PRIVATE trtmc_model_wan_t2v)
  endif()
  trtmc_add_test(test_wan_generation_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_generation_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_wan_generation_conditioning
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_generation_conditioning.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_wan_denoising_step_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_denoising_step_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_wan_gpu_matmul
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan_gpu_matmul.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_wan_gpu_matmul)
    target_link_libraries(test_wan_gpu_matmul PRIVATE trtmc_model_wan_t2v)
  endif()
endif()
