# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_dinov3 SHARED
  image_preprocess.cpp
  pipeline.cpp
  plugin.cpp
  ${TRTMC_MODEL_dinov3_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_dinov3 PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_dinov3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_dinov3 PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_dinov3 PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_dinov3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_dinov3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/dinov3"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_dinov3_image_preprocess
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_dinov3_image_preprocess.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_dinov3_image_preprocess)
    target_link_libraries(test_dinov3_image_preprocess PRIVATE trtmc_model_dinov3)
  endif()
  trtmc_add_test(test_dinov3_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_dinov3_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_dinov3_pipeline)
    target_link_libraries(test_dinov3_pipeline PRIVATE trtmc_model_dinov3)
  endif()
endif()
