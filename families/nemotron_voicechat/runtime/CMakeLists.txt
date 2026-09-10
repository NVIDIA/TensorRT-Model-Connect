# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_nemotron_voicechat_live_control INTERFACE)
target_include_directories(trtmc_nemotron_voicechat_live_control INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
install(TARGETS trtmc_nemotron_voicechat_live_control EXPORT trtmcTargets)
install(FILES ../include/trtmc/nemotron_voicechat/live_control.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc/nemotron_voicechat
)

add_library(trtmc_model_nemotron_voicechat SHARED
  bpe_tokenizer.cpp
  resampler.cpp
  audio_helpers.cpp
  codec_reconstruction.cpp
  conversation_memory.cpp
  function_channel.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  session_state.cpp
  thinker_hybrid_state.cpp
  thinker_kv_cache.cpp
  thinker_mamba_state.cpp
)

target_include_directories(trtmc_model_nemotron_voicechat
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_nemotron_voicechat SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_nemotron_voicechat PRIVATE
  trtmc_nemotron_voicechat_live_control
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_nemotron_voicechat PRIVATE
  ${TRTMC_CXX_WARNINGS}
)
set_target_properties(trtmc_model_nemotron_voicechat PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron_voicechat
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TRTMC_BUILD_TESTS)
  find_package(Threads REQUIRED)
  if(TRTMC_BUILD_CLI)
    add_executable(test_nemotron_voicechat_lifecycle_probe_host
      ../tests/cpp/native_lifecycle_probe.cpp
    )
    target_include_directories(test_nemotron_voicechat_lifecycle_probe_host PRIVATE
      ${PROJECT_SOURCE_DIR}/apps
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(test_nemotron_voicechat_lifecycle_probe_host PRIVATE
      trtmc_cli
      Threads::Threads
    )
    target_compile_options(test_nemotron_voicechat_lifecycle_probe_host PRIVATE
      ${TRTMC_CXX_WARNINGS}
    )
    set_target_properties(test_nemotron_voicechat_lifecycle_probe_host PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      BUILD_RPATH "\$ORIGIN"
    )
    add_test(NAME nemotron_voicechat_lifecycle_probe_host
      COMMAND test_nemotron_voicechat_lifecycle_probe_host
    )
  endif()

  foreach(test_name IN ITEMS
      test_nemotron_voicechat_codec_reconstruction
      test_nemotron_voicechat_conversation_memory
      test_nemotron_voicechat_function_channel
      test_nemotron_voicechat_session_state
      test_nemotron_voicechat_streaming_mel_policy)
    string(REPLACE "test_nemotron_voicechat_" "" test_source ${test_name})
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/nemotron_voicechat/tests/cpp/test_${test_source}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(${test_name} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_nemotron_voicechat
      Threads::Threads
    )
    target_compile_options(${test_name} PRIVATE
      ${TRTMC_CXX_WARNINGS}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
