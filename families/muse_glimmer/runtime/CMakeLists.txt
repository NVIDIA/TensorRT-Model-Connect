# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)

add_library(trtmc_model_muse_glimmer SHARED plugin.cpp native_pipeline.cpp tokenizer.cpp sampler.cpp)
target_include_directories(trtmc_model_muse_glimmer PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_muse_glimmer SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_muse_glimmer PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ICU::uc
  ICU::i18n
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_muse_glimmer PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_muse_glimmer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_muse_glimmer LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
include(edge_llm/Adapter.cmake)
