# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_vibevoice_asr SHARED
  audio_frontend.cpp
  bpe_tokenizer.cpp
  host_plan.cpp
  pipeline.cpp
  plugin.cpp
  prompt.cpp
  runtime_config.cpp
  transcript.cpp
)
target_include_directories(trtmc_model_vibevoice_asr PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_vibevoice_asr SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_vibevoice_asr PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_vibevoice_asr PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_vibevoice_asr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_vibevoice_asr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  # Host-only front end, prompt and transcript tests: no GPU, no engines.
  add_executable(test_vibevoice_asr_host
    ${PROJECT_SOURCE_DIR}/families/vibevoice_asr/tests/cpp/test_host.cpp
    audio_frontend.cpp
    bpe_tokenizer.cpp
    host_plan.cpp
    prompt.cpp
    runtime_config.cpp
    transcript.cpp
  )
  target_include_directories(test_vibevoice_asr_host PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_vibevoice_asr_host PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_vibevoice_asr_host PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME vibevoice_asr_host COMMAND test_vibevoice_asr_host)
endif()
