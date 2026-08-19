# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_sam SHARED
  plugin.cpp
  plugin_helpers.cpp
  sam_pipeline.cpp
  ${TRTMC_MODEL_sam_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_sam PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_sam SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_sam PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_sam PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_sam PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sam PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/sam"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_sam_prompt_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam_prompt_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_sam_prompt_seam)
    target_link_libraries(test_sam_prompt_seam PRIVATE trtmc_model_sam)
  endif()
  trtmc_add_test(test_sam_image_preprocess_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam_image_preprocess_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_sam_image_preprocess_seam)
    target_link_libraries(test_sam_image_preprocess_seam PRIVATE trtmc_model_sam)
  endif()
  trtmc_add_test(test_sam_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_sam_pipeline)
    target_link_libraries(test_sam_pipeline PRIVATE trtmc_model_sam)
  endif()
endif()
