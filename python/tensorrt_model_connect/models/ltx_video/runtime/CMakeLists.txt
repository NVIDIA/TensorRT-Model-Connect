# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_ltx_video SHARED
  diffusion_helpers.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_ltx_video_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_ltx_video PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_ltx_video SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_ltx_video PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_ltx_video PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_ltx_video PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_ltx_video PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/ltx_video"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_ltx_video_scheduler
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_ltx_video_scheduler.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_ltx_video_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_ltx_video_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_ltx_video_pipeline)
    target_link_libraries(test_ltx_video_pipeline PRIVATE trtmc_model_ltx_video)
    target_sources(test_ltx_video_pipeline PRIVATE
      plugin_helpers.cpp
      diffusion_helpers.cpp
    )
  endif()
endif()
