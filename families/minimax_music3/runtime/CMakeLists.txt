# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_music3 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  prompt_format.cpp
)

target_include_directories(trtmc_model_minimax_music3 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minimax_music3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_minimax_music3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_minimax_music3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_minimax_music3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minimax_music3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_minimax_music3_prompt_format
    ${PROJECT_SOURCE_DIR}/families/minimax_music3/tests/cpp/test_minimax_music3_prompt_format.cpp
  )
  target_include_directories(test_minimax_music3_prompt_format PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_minimax_music3_prompt_format PRIVATE
    trtmc_model_minimax_music3
  )
  target_compile_options(test_minimax_music3_prompt_format PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_music3_prompt_format COMMAND test_minimax_music3_prompt_format)
endif()
