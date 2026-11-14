# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_blip SHARED
  image_preprocess.cpp
  wordpiece_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_blip PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_blip SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_blip PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_blip PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_blip PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_blip LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_subdirectory("${PROJECT_SOURCE_DIR}/families/blip/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
