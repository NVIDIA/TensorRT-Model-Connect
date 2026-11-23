# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(lighton_ocr_test_image_io STATIC image_io.cpp)
target_include_directories(lighton_ocr_test_image_io PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(lighton_ocr_test_image_io SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)

add_executable(test_lighton_ocr_diagnostic diagnostic.cpp)
target_include_directories(test_lighton_ocr_diagnostic PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(test_lighton_ocr_diagnostic SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(test_lighton_ocr_diagnostic PRIVATE
  trtmc_model_lighton_ocr trtmc_runtime lighton_ocr_test_image_io nlohmann_json::nlohmann_json)

add_executable(test_lighton_ocr_sdk_cpp sdk_consumer.cpp)
target_include_directories(test_lighton_ocr_sdk_cpp PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/api/include)
target_link_libraries(test_lighton_ocr_sdk_cpp PRIVATE trtmc_c lighton_ocr_test_image_io nlohmann_json::nlohmann_json)
add_executable(test_lighton_ocr_sdk_c sdk_consumer.c)
target_include_directories(test_lighton_ocr_sdk_c PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/api/include)
target_link_libraries(test_lighton_ocr_sdk_c PRIVATE trtmc_c lighton_ocr_test_image_io)
add_dependencies(trtmc_model_lighton_ocr test_lighton_ocr_sdk_cpp test_lighton_ocr_sdk_c)

add_executable(test_lighton_ocr_runtime test_runtime.cpp)
target_include_directories(test_lighton_ocr_runtime PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(test_lighton_ocr_runtime SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(test_lighton_ocr_runtime PRIVATE trtmc_model_lighton_ocr trtmc_core)
add_dependencies(test_lighton_ocr_runtime test_lighton_ocr_diagnostic)
add_test(NAME test_lighton_ocr_runtime COMMAND test_lighton_ocr_runtime)

set_target_properties(test_lighton_ocr_diagnostic test_lighton_ocr_sdk_cpp test_lighton_ocr_sdk_c
  test_lighton_ocr_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")
