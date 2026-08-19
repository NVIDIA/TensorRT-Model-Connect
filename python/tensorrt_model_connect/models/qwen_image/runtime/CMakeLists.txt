# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen_image SHARED
  diffusion_helpers.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  qwen_image_scheduler.cpp
  qwen_image_types.cpp
  ${TRTMC_MODEL_qwen_image_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_qwen_image PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_qwen_image SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_qwen_image PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_qwen_image PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_qwen_image PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen_image PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/qwen_image"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_qwen_image_cfg_renorm
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_image_cfg_renorm.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_qwen_image_cfg_renorm)
    target_link_libraries(test_qwen_image_cfg_renorm PRIVATE trtmc_model_qwen_image)
  endif()
  trtmc_add_test(test_qwen_image_flow_match_scheduler
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_image_flow_match_scheduler.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_qwen_image_flow_match_scheduler)
    target_sources(test_qwen_image_flow_match_scheduler PRIVATE
      qwen_image_scheduler.cpp
    )
  endif()
  trtmc_add_test(test_qwen_image_batch_utils
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen_image_batch_utils.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
endif()
