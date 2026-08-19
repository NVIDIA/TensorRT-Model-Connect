# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_whisper SHARED
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  whisper_mel_spectrogram.cpp
  ${TRTMC_MODEL_whisper_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_whisper PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_whisper SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_whisper PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_whisper PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_whisper PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_whisper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/whisper"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_whisper_mel_spectrogram
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_whisper_mel_spectrogram.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_whisper_mel_spectrogram)
    target_sources(test_whisper_mel_spectrogram PRIVATE
      whisper_mel_spectrogram.cpp
    )
  endif()
  trtmc_add_test(test_whisper_host_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_whisper_host_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_whisper_decode_policy
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_whisper_decode_policy.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_whisper_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_whisper_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
  )
  if(TARGET test_whisper_pipeline)
    target_link_libraries(test_whisper_pipeline PRIVATE trtmc_model_whisper)
  endif()
endif()
