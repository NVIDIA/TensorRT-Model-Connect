# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_pixart SHARED
  diffusion_helpers.cpp
  gpu_matmul.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_pixart_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_pixart PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_pixart SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
if(NOT TRTMC_CUBLAS_LIBRARY)
  message(FATAL_ERROR "Model 'pixart' requires cuBLAS, but it was not found")
endif()
target_link_libraries(trtmc_model_pixart PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${TRTMC_CUBLAS_LIBRARY}
)
target_compile_definitions(trtmc_model_pixart PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_pixart PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_pixart PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/pixart"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
set_source_files_properties(pipeline.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_pixart_denoising_step_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_pixart_denoising_step_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_pixart_gpu_matmul
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_pixart_gpu_matmul.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_pixart_gpu_matmul)
    target_link_libraries(test_pixart_gpu_matmul PRIVATE trtmc_model_pixart)
  endif()
endif()
