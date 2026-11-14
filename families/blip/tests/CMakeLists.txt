# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_blip_native cpp/test_native.cpp)
set_target_properties(test_blip_native PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(test_blip_native PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(test_blip_native SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
target_link_libraries(test_blip_native PRIVATE
  trtmc_model_blip
  trtmc_runtime
  nlohmann_json::nlohmann_json
)
target_compile_options(test_blip_native PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_blip_tokenizer cpp/test_tokenizer.cpp ../runtime/wordpiece_tokenizer.cpp)
set_target_properties(test_blip_tokenizer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")
target_include_directories(test_blip_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_blip_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc)
target_compile_options(test_blip_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_blip_tokenizer COMMAND test_blip_tokenizer)
set_tests_properties(test_blip_tokenizer PROPERTIES TIMEOUT 15)
