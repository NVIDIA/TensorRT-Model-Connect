# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_magpie_tts SHARED
  audio_helpers.cpp
  config_schema.cpp
  cuda_common.cpp
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_magpie_tts_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_magpie_tts PRIVATE
    magpie_kernels.cu
  )
endif()
target_include_directories(trtmc_model_magpie_tts PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_magpie_tts SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_magpie_tts PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_magpie_tts PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_magpie_tts PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_magpie_tts PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/magpie_tts"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
set_source_files_properties(pipeline.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_magpie_config_schema
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_config_schema.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_magpie_config_schema)
    target_link_libraries(test_magpie_config_schema PRIVATE trtmc_model_magpie_tts)
  endif()
  trtmc_add_test(test_magpie_decode_policy
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_decode_policy.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_magpie_decoder_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_decoder_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_magpie_generation_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_generation_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_magpie_text_completion_policy
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_text_completion_policy.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_magpie_codec_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_codec_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_magpie_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_magpie_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_magpie_pipeline)
    target_link_libraries(test_magpie_pipeline PRIVATE trtmc_model_magpie_tts)
  endif()
endif()
