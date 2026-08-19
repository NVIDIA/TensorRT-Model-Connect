# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_personaplex SHARED
  audio_helpers.cpp
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_personaplex_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_personaplex PRIVATE
    sampling_kernels.cu
  )
endif()
target_include_directories(trtmc_model_personaplex PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_personaplex SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_personaplex PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_personaplex PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_personaplex PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_personaplex PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/personaplex"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
set_source_files_properties(pipeline.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_personaplex_speech_subprocess_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_subprocess_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_decode_stop_policy
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_decode_stop_policy.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_generation_helpers
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_generation_helpers.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_depth_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_depth_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_runtime_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_runtime_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_performance
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_performance.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_temporal_embed_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_temporal_embed_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_speech_mimi_decode_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_mimi_decode_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_personaplex_decode_runtime
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_decode_runtime.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_personaplex_decode_runtime)
    target_sources(test_personaplex_decode_runtime PRIVATE
      decode_runtime.cpp
    )
  endif()
  trtmc_add_test(test_personaplex_sampling_kernels
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_sampling_kernels.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_personaplex_sampling_kernels)
    target_link_libraries(test_personaplex_sampling_kernels PRIVATE trtmc_model_personaplex)
  endif()
  trtmc_add_test(test_personaplex_speech_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_personaplex_speech_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_personaplex_speech_pipeline)
    target_link_libraries(test_personaplex_speech_pipeline PRIVATE trtmc_model_personaplex)
  endif()
endif()
