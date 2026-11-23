# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_nemotron_bolt SHARED plugin.cpp native_task.cpp native_language.cpp native_proposal.cpp native_lilicorr.cpp
  native_vision.cpp visual_frontend.cpp media_geometry.cpp tokenizer.cpp chat.cpp sampler.cpp)
target_include_directories(trtmc_model_nemotron_bolt SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_include_directories(trtmc_model_nemotron_bolt PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_model_nemotron_bolt PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
set_target_properties(trtmc_model_nemotron_bolt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
include("${CMAKE_CURRENT_LIST_DIR}/edge_llm/Adapter.cmake")
install(TARGETS trtmc_model_nemotron_bolt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_nemotron_bolt_sdk_cpp ../tests/sdk_consumer.cpp)
  target_link_libraries(test_nemotron_bolt_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_nemotron_bolt_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
