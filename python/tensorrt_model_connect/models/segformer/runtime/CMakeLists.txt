# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_segformer SHARED
  plugin.cpp
  plugin_helpers.cpp
  segformer_preprocess_seam.cpp
  segment_pipeline.cpp
  ${TRTMC_MODEL_segformer_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_segformer PRIVATE
    segformer_postprocess_cuda.cu
  )
endif()
target_include_directories(trtmc_model_segformer PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_segformer SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_segformer PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_segformer PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_segformer PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_segformer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/segformer"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_segformer_preprocess_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_segformer_preprocess_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_segformer_preprocess_seam)
    target_link_libraries(test_segformer_preprocess_seam PRIVATE trtmc_model_segformer)
  endif()
  trtmc_add_test(test_segformer_postprocess_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_segformer_postprocess_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_segformer_segment_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_segformer_segment_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_segformer_segment_pipeline)
    target_link_libraries(test_segformer_segment_pipeline PRIVATE trtmc_model_segformer)
  endif()
endif()
