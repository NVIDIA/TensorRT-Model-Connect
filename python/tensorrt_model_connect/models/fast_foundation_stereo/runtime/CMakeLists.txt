# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_fast_foundation_stereo SHARED
  plugin.cpp
  stereo_pipeline.cpp
  ${TRTMC_MODEL_fast_foundation_stereo_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_fast_foundation_stereo PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_fast_foundation_stereo SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_fast_foundation_stereo PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_definitions(trtmc_model_fast_foundation_stereo PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_fast_foundation_stereo PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_fast_foundation_stereo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/fast_foundation_stereo"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_fast_foundation_stereo_preprocess
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_fast_foundation_stereo_preprocess.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_fast_foundation_stereo_preprocess)
    target_link_libraries(test_fast_foundation_stereo_preprocess PRIVATE
      trtmc_model_fast_foundation_stereo)
  endif()
endif()
