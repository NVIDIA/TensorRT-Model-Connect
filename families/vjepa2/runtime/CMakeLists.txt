# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_vjepa2 SHARED preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_vjepa2 PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_vjepa2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_vjepa2 PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_vjepa2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_vjepa2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_vjepa2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_vjepa2_preprocess ${PROJECT_SOURCE_DIR}/families/vjepa2/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_vjepa2_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_vjepa2_preprocess PRIVATE trtmc_model_vjepa2 nlohmann_json::nlohmann_json)
  add_executable(test_vjepa2_runtime ${PROJECT_SOURCE_DIR}/families/vjepa2/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_vjepa2_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_vjepa2_preprocess test_vjepa2_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
