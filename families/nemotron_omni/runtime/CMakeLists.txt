# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_nemotron_omni SHARED
  plugin.cpp native_decoder.cpp native_media.cpp audio_frontend.cpp
  visual_frontend.cpp media_geometry.cpp resampler.cpp tokenizer.cpp chat.cpp sampler.cpp)
target_include_directories(trtmc_model_nemotron_omni PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_nemotron_omni SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_nemotron_omni PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_nemotron_omni PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_nemotron_omni PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_nemotron_omni LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

include("${CMAKE_CURRENT_LIST_DIR}/edge_llm/Adapter.cmake")

if(TRTMC_BUILD_TESTS)
  add_executable(nemotron_omni_inference_test ../tests/cpp/run_model.cpp)
  target_link_libraries(nemotron_omni_inference_test PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  add_executable(nemotron_omni_decoder_probe
    ../tests/cpp/decoder_probe.cpp native_decoder.cpp)
  target_include_directories(nemotron_omni_decoder_probe PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_link_libraries(nemotron_omni_decoder_probe PRIVATE
    trtmc_core trtmc_backend_trt nlohmann_json::nlohmann_json)
  add_dependencies(trtmc_model_nemotron_omni
    nemotron_omni_inference_test nemotron_omni_decoder_probe)
endif()
