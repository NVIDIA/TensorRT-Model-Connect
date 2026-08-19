# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_z_image SHARED
  diffusion_helpers.cpp
  gpu_matmul.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_z_image_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_z_image PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_z_image SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
if(NOT TRTMC_CUBLAS_LIBRARY)
  message(FATAL_ERROR "Model 'z_image' requires cuBLAS, but it was not found")
endif()
target_link_libraries(trtmc_model_z_image PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${TRTMC_CUBLAS_LIBRARY}
)
target_compile_definitions(trtmc_model_z_image PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_z_image PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_z_image PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/z_image"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_z_image_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_z_image_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_z_image_pipeline)
    target_link_libraries(test_z_image_pipeline PRIVATE trtmc_model_z_image)
  endif()
  trtmc_add_test(test_z_image_batch_utils
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_z_image_batch_utils.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_z_image_diffusion_math
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_z_image_diffusion_math.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_z_image_gpu_matmul
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_z_image_gpu_matmul.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_z_image_gpu_matmul)
    target_link_libraries(test_z_image_gpu_matmul PRIVATE trtmc_model_z_image)
  endif()
endif()
