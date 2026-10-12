# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_birefnet SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_birefnet
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_birefnet SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_birefnet PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_birefnet PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_birefnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_birefnet_preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp/test_preprocess.cpp
    image_preprocess_seam.cpp
  )
  target_include_directories(test_birefnet_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  add_test(NAME birefnet_preprocess COMMAND test_birefnet_preprocess)
endif()
