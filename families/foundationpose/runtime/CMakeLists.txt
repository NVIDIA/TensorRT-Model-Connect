# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_foundationpose SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_foundationpose PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_foundationpose SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_foundationpose PRIVATE
  trtmc_core
)
target_compile_options(trtmc_model_foundationpose PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_foundationpose PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_foundationpose
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(foundationpose_qualification
    ${PROJECT_SOURCE_DIR}/families/foundationpose/tests/cpp/qualification.cpp
  )
  target_include_directories(foundationpose_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(foundationpose_qualification SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(foundationpose_qualification PRIVATE
    trtmc_runtime
    ${TRTMC_CUDART_LIBRARY}
  )
  add_dependencies(foundationpose_qualification
    trtmc_backend_trt
    trtmc_model_foundationpose
  )
  target_compile_options(foundationpose_qualification PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(test_foundationpose_pipeline
    ${PROJECT_SOURCE_DIR}/families/foundationpose/tests/cpp/test_pipeline.cpp
  )
  target_include_directories(test_foundationpose_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_foundationpose_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_foundationpose_pipeline PRIVATE
    trtmc_model_foundationpose
  )
  add_dependencies(test_foundationpose_pipeline foundationpose_qualification)
  target_compile_options(test_foundationpose_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME foundationpose_pipeline COMMAND test_foundationpose_pipeline)
endif()
