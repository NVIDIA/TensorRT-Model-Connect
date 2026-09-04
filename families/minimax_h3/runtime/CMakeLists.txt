# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_h3 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  torch_cuda_normal.cu
)

set_source_files_properties(
  torch_cuda_normal.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_minimax_h3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minimax_h3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_minimax_h3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_minimax_h3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_minimax_h3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minimax_h3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_minimax_h3_math
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_minimax_h3_math.cpp
  )
  target_include_directories(test_minimax_h3_math PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_math SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_minimax_h3_math PRIVATE trtmc_model_minimax_h3)
  target_compile_options(test_minimax_h3_math PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_math COMMAND test_minimax_h3_math)
endif()
