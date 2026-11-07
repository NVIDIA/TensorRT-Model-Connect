# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lfm2_audio SHARED
  audio_dsp.cpp
  bpe_tokenizer.cpp
  byte_level_decoder.cpp
  conv_state.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  pretokenizer.cpp
)
target_include_directories(trtmc_model_lfm2_audio PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm2_audio SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_lfm2_audio PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lfm2_audio PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_lfm2_audio PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm2_audio LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(lfm2_audio_api_probe ${PROJECT_SOURCE_DIR}/families/lfm2_audio/tests/cpp/api_probe.cpp)
  target_include_directories(lfm2_audio_api_probe PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  target_link_libraries(lfm2_audio_api_probe PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  add_executable(lfm2_audio_probe ${PROJECT_SOURCE_DIR}/families/lfm2_audio/tests/cpp/audio_probe.cpp)
  target_include_directories(lfm2_audio_probe PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(lfm2_audio_probe PRIVATE trtmc_model_lfm2_audio trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
  add_executable(lfm2_audio_dsp_probe
    ${PROJECT_SOURCE_DIR}/families/lfm2_audio/tests/cpp/dsp_probe.cpp
    audio_dsp.cpp
  )
  target_include_directories(lfm2_audio_dsp_probe PRIVATE ${PROJECT_SOURCE_DIR})
endif()
