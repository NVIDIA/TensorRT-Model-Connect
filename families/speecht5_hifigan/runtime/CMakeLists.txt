# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_speecht5_hifigan SHARED plugin.cpp)
target_include_directories(trtmc_model_speecht5_hifigan PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_speecht5_hifigan SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_speecht5_hifigan PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_speecht5_hifigan PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_speecht5_hifigan PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_speecht5_hifigan LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_speecht5_hifigan_runtime ${PROJECT_SOURCE_DIR}/families/speecht5_hifigan/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_speecht5_hifigan_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_speecht5_hifigan_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
