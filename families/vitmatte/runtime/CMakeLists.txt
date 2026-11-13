# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_vitmatte SHARED plugin.cpp)
target_include_directories(trtmc_model_vitmatte PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_vitmatte SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_vitmatte PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_vitmatte PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_vitmatte PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_vitmatte LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_vitmatte_runtime ${PROJECT_SOURCE_DIR}/families/vitmatte/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_vitmatte_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_vitmatte_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
