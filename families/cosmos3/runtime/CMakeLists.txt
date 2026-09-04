# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_cosmos3 SHARED
  bpe_tokenizer.cpp
  conditioning.cpp
  cosmos3_unipc_cuda.cu
  distributed_runtime.cpp
  pipeline.cpp
  plugin.cpp
  runtime_config.cpp
  torch_cuda_normal.cu
  vae_cache_storage.cpp
)

set_source_files_properties(
  cosmos3_unipc_cuda.cu
  torch_cuda_normal.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_cosmos3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_cosmos3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_cosmos3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_cosmos3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_cosmos3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_cosmos3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(test_name IN ITEMS conditioning runtime_config vae_cache)
    add_executable(test_cosmos3_${test_name}
      ${PROJECT_SOURCE_DIR}/families/cosmos3/tests/cpp/test_cosmos3_${test_name}.cpp
    )
    target_include_directories(test_cosmos3_${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_cosmos3_${test_name} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_cosmos3_${test_name} PRIVATE trtmc_model_cosmos3)
    target_compile_options(test_cosmos3_${test_name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME cosmos3_${test_name} COMMAND test_cosmos3_${test_name})
  endforeach()
endif()
