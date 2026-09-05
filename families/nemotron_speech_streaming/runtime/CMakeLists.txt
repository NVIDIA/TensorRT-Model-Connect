# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_speech_streaming SHARED
  bpe_tokenizer.cpp
  distributed_runtime.cpp
  unigram_tokenizer.cpp
  resampler.cpp
  audio_helpers.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_nemotron_speech_streaming
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_nemotron_speech_streaming SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_nemotron_speech_streaming PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_nemotron_speech_streaming PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_nemotron_speech_streaming PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron_speech_streaming
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_nemotron_speech_streaming_tokenizer
    ${PROJECT_SOURCE_DIR}/families/nemotron_speech_streaming/tests/cpp/test_tokenizer.cpp
  )
  target_include_directories(test_nemotron_speech_streaming_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_nemotron_speech_streaming_tokenizer PRIVATE
    trtmc_model_nemotron_speech_streaming
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME nemotron_speech_streaming_tokenizer
    COMMAND test_nemotron_speech_streaming_tokenizer)

  foreach(test_name IN ITEMS
      test_nemotron_speech_streaming_audio_helpers
      test_nemotron_speech_streaming_decode_policy
      test_nemotron_speech_streaming_streaming_contract)
    string(REPLACE "test_nemotron_speech_streaming_" "" test_source ${test_name})
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/nemotron_speech_streaming/tests/cpp/test_${test_source}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(${test_name} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_nemotron_speech_streaming
    )
    target_compile_options(${test_name} PRIVATE
      -Wall -Wextra -Wpedantic
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
