# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hstu SHARED pipeline.cpp config.cpp plugin.cpp)
target_include_directories(trtmc_model_hstu PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_hstu SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_hstu PRIVATE trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_hstu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_hstu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(trtmc_hstu runner.cpp)
target_include_directories(trtmc_hstu PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_hstu PRIVATE trtmc_runtime nlohmann_json::nlohmann_json)
target_compile_options(trtmc_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_hstu PROPERTIES OUTPUT_NAME trtmc-hstu
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS trtmc_hstu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# A selectively built family includes its public native command.
add_dependencies(trtmc_model_hstu trtmc_hstu)

if(TRTMC_BUILD_TESTS)
  add_executable(test_hstu_pipeline ../tests/cpp/test_pipeline.cpp)
  target_include_directories(test_hstu_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_pipeline PRIVATE trtmc_model_hstu)
  target_compile_options(test_hstu_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_pipeline COMMAND test_hstu_pipeline)
  set_tests_properties(hstu_pipeline PROPERTIES LABELS cpu)
endif()
