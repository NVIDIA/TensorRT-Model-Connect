# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU COMPONENTS uc QUIET)
if(NOT ICU_FOUND)
  message(STATUS "Skipping blip2: install ICU development packages for Unicode tokenization")
  return()
endif()
add_library(trtmc_model_blip2 SHARED bpe_tokenizer.cpp image_preprocess_seam.cpp plugin.cpp)
target_include_directories(trtmc_model_blip2 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_blip2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_blip2 PRIVATE ICU::uc trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_blip2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_blip2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_blip2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_blip2_preprocess ${PROJECT_SOURCE_DIR}/families/blip2/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_blip2_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_blip2_preprocess PRIVATE trtmc_model_blip2 nlohmann_json::nlohmann_json)
  set_target_properties(test_blip2_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_blip2_image_resize ${PROJECT_SOURCE_DIR}/families/blip2/tests/cpp/test_image_resize.cpp)
  target_include_directories(test_blip2_image_resize PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_blip2_image_resize PRIVATE trtmc_model_blip2)
  add_test(NAME blip2_image_resize COMMAND test_blip2_image_resize)
endif()
