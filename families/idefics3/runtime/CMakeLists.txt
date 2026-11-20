# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_idefics3 SHARED bpe_tokenizer.cpp preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_idefics3 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_idefics3 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_idefics3 PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_idefics3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_idefics3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_idefics3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_idefics3_preprocess ${PROJECT_SOURCE_DIR}/families/idefics3/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_idefics3_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_idefics3_preprocess PRIVATE trtmc_model_idefics3 nlohmann_json::nlohmann_json)
  set_target_properties(test_idefics3_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
