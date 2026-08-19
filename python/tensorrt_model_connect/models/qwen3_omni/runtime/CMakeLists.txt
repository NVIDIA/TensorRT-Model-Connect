# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_omni SHARED
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  talker_runtime.cpp
  ${TRTMC_MODEL_qwen3_omni_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_qwen3_omni PRIVATE
    argmax_kernel.cu
  )
endif()
target_include_directories(trtmc_model_qwen3_omni PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_qwen3_omni SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_qwen3_omni PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_qwen3_omni PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_qwen3_omni PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen3_omni PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/qwen3_omni"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_qwen3_omni_audio_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen3_omni_audio_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_qwen3_omni_talker_runtime
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen3_omni_talker_runtime.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_qwen3_omni_talker_runtime)
    target_sources(test_qwen3_omni_talker_runtime PRIVATE
      talker_runtime.cpp
    )
  endif()
  trtmc_add_test(test_qwen3_omni_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_qwen3_omni_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_qwen3_omni_pipeline)
    target_link_libraries(test_qwen3_omni_pipeline PRIVATE trtmc_model_qwen3_omni)
  endif()
endif()
