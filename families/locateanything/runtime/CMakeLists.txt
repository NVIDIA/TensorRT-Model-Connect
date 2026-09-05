# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_locateanything SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  image_preprocessor.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

set_source_files_properties(
  PROPERTIES LANGUAGE CUDA
)
set_source_files_properties(image_preprocessor.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)

target_include_directories(trtmc_model_locateanything
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_locateanything SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_locateanything PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_locateanything PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_locateanything PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_locateanything
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
