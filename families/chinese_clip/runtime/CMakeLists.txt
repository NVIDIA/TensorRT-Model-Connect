# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU COMPONENTS uc QUIET)
if(NOT ICU_FOUND)
  message(STATUS "Skipping chinese_clip: install ICU development packages for Unicode tokenization")
  return()
endif()
add_library(trtmc_model_chinese_clip SHARED
  wordpiece_tokenizer.cpp
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_chinese_clip
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_chinese_clip SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_chinese_clip PRIVATE
  trtmc_core
  ICU::uc
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_chinese_clip PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_chinese_clip PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_chinese_clip
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_chinese_clip_image_preprocess_seam
    ${PROJECT_SOURCE_DIR}/families/chinese_clip/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_chinese_clip_image_preprocess_seam PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_chinese_clip_image_preprocess_seam PRIVATE
    trtmc_model_chinese_clip
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_chinese_clip_image_preprocess_seam PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME chinese_clip_image_preprocess_seam
    COMMAND test_chinese_clip_image_preprocess_seam
  )
endif()
