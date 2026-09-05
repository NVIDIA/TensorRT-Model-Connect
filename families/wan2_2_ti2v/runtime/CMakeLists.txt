# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wan2_2_ti2v SHARED
  unigram_tokenizer.cpp
  easycache.cpp
  options.cpp
  pipeline.cpp
  plugin.cpp
  prompt_cleaner.cpp
  runtime_config.cpp
  vae_cache_storage.cpp
  torch_cuda_normal.cu
  wan2_2_unipc_cuda.cu
)

set_source_files_properties(
  torch_cuda_normal.cu
  wan2_2_unipc_cuda.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_wan2_2_ti2v
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_wan2_2_ti2v SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_wan2_2_ti2v PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_wan2_2_ti2v PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_wan2_2_ti2v PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_wan2_2_ti2v
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
