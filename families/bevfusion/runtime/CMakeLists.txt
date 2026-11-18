# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_bevfusion SHARED pipeline.cpp plugin.cpp)
target_include_directories(trtmc_model_bevfusion PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_bevfusion SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_bevfusion PRIVATE trtmc_core ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_bevfusion PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_bevfusion PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_bevfusion LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_bevfusion_pipeline
    ${PROJECT_SOURCE_DIR}/families/bevfusion/tests/cpp/test_pipeline.cpp)
  target_include_directories(test_bevfusion_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_bevfusion_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bevfusion_pipeline PRIVATE trtmc_model_bevfusion trtmc_core)
  target_compile_options(test_bevfusion_pipeline PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME bevfusion_pipeline COMMAND test_bevfusion_pipeline)
  set_tests_properties(bevfusion_pipeline PROPERTIES LABELS "cpu")
  add_executable(test_bevfusion_sdk_cpp ${PROJECT_SOURCE_DIR}/families/bevfusion/tests/sdk_consumer.cpp)
  target_include_directories(test_bevfusion_sdk_cpp PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  target_include_directories(test_bevfusion_sdk_cpp SYSTEM PRIVATE ${TRTMC_TRT_INCLUDE_DIR} ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_bevfusion_sdk_cpp PRIVATE trtmc_c trtmc_core nlohmann_json::nlohmann_json ${TRTMC_TRT_LIBRARY} ${TRTMC_CUDART_LIBRARY})
  target_compile_options(test_bevfusion_sdk_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_bevfusion_sdk_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_bevfusion_sdk_c ${PROJECT_SOURCE_DIR}/families/bevfusion/tests/sdk_consumer.c)
  target_include_directories(test_bevfusion_sdk_c PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  target_link_libraries(test_bevfusion_sdk_c PRIVATE trtmc_c)
  target_compile_options(test_bevfusion_sdk_c PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_bevfusion_sdk_c PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_dependencies(test_bevfusion_pipeline test_bevfusion_sdk_c test_bevfusion_sdk_cpp)
endif()
