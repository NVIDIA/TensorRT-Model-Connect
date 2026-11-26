# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_higgs_audio_v2_tokenizer SHARED plugin.cpp)
target_include_directories(trtmc_model_higgs_audio_v2_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_higgs_audio_v2_tokenizer SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_higgs_audio_v2_tokenizer PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_higgs_audio_v2_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_higgs_audio_v2_tokenizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_higgs_audio_v2_tokenizer LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_higgs_audio_v2_tokenizer_runtime ${PROJECT_SOURCE_DIR}/families/higgs_audio_v2_tokenizer/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_higgs_audio_v2_tokenizer_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_higgs_audio_v2_tokenizer_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
