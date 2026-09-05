# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_elf_flow SHARED
  unigram_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampling_kernels.cu
)

set_source_files_properties(
  sampling_kernels.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_elf_flow
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_elf_flow SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_elf_flow PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_elf_flow PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_elf_flow PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_elf_flow
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_elf_flow_runtime_config
    ${PROJECT_SOURCE_DIR}/families/elf_flow/tests/cpp/test_elf_flow_runtime_config.cpp
  )
  target_include_directories(test_elf_flow_runtime_config PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_elf_flow_runtime_config PRIVATE trtmc_model_elf_flow)
  target_compile_options(test_elf_flow_runtime_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME elf_flow_runtime_config COMMAND test_elf_flow_runtime_config)
endif()
