# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_flint SHARED plugin.cpp pipeline.cpp bpe_tokenizer.cpp sampler.cpp)
target_include_directories(trtmc_model_flint PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_flint SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_flint PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
target_compile_options(trtmc_model_flint PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_flint PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_flint LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(flint_logits_trace ../tests/cpp/logits_trace.cpp)
  target_include_directories(flint_logits_trace PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(flint_logits_trace PRIVATE
    trtmc_runtime trtmc_model_flint nlohmann_json::nlohmann_json)
  set_target_properties(flint_logits_trace PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
