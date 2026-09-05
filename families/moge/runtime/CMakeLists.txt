# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_moge SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_moge PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_link_libraries(trtmc_model_moge PRIVATE trtmc_core)
target_compile_options(trtmc_model_moge PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_moge PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_moge LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_moge_pipeline
    ${PROJECT_SOURCE_DIR}/families/moge/tests/cpp/test_moge_pipeline.cpp
  )
  target_include_directories(test_moge_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_moge_pipeline PRIVATE trtmc_model_moge trtmc_core)
  target_compile_options(test_moge_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_moge_pipeline COMMAND test_moge_pipeline)
endif()
