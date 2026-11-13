# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_paddleocr_vl SHARED bpe_tokenizer.cpp preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_paddleocr_vl PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_paddleocr_vl SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_paddleocr_vl PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_paddleocr_vl PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_paddleocr_vl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_paddleocr_vl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_paddleocr_vl_preprocess ${PROJECT_SOURCE_DIR}/families/paddleocr_vl/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_paddleocr_vl_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_paddleocr_vl_preprocess PRIVATE trtmc_model_paddleocr_vl nlohmann_json::nlohmann_json)
  set_target_properties(test_paddleocr_vl_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
