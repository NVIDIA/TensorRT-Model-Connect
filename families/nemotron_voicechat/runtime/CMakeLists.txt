# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_voicechat SHARED
  bpe_tokenizer.cpp
  resampler.cpp
  audio_helpers.cpp
  codec_reconstruction.cpp
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
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_nemotron_voicechat PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_nemotron_voicechat PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron_voicechat
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  find_package(Threads REQUIRED)
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
    -Wall -Wextra -Wpedantic
  )
  set_target_properties(test_nemotron_voicechat_lifecycle_probe_host PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN"
  )
  add_test(NAME nemotron_voicechat_lifecycle_probe_host
    COMMAND test_nemotron_voicechat_lifecycle_probe_host
  )
endif()
