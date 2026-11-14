# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lighton_ocr SHARED image.cpp tokenizer.cpp pipeline.cpp plugin.cpp)
target_include_directories(trtmc_model_lighton_ocr PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_lighton_ocr SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_lighton_ocr PRIVATE trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_lighton_ocr PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_lighton_ocr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_lighton_ocr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
