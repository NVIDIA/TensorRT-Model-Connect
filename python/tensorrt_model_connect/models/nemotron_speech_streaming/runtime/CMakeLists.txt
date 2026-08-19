# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_speech_streaming SHARED
  audio_helpers.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_nemotron_speech_streaming_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_nemotron_speech_streaming PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_nemotron_speech_streaming SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_nemotron_speech_streaming PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_nemotron_speech_streaming PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_nemotron_speech_streaming PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_nemotron_speech_streaming PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/nemotron_speech_streaming"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_nemotron_speech_streaming_audio_helpers
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_speech_streaming_audio_helpers.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_nemotron_speech_streaming_audio_helpers)
    target_sources(test_nemotron_speech_streaming_audio_helpers PRIVATE
      audio_helpers.cpp
    )
  endif()
  trtmc_add_test(test_nemotron_speech_streaming_decode_policy
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_speech_streaming_decode_policy.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_nemotron_speech_streaming_streaming_contract
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_nemotron_speech_streaming_streaming_contract.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
endif()
