# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_ast SHARED
  fbank.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_ast
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_ast SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_ast PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_ast PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_ast PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_ast
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_ast_fbank
    ${PROJECT_SOURCE_DIR}/families/ast/tests/cpp/test_fbank.cpp
  )
  target_include_directories(test_ast_fbank PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_ast_fbank PRIVATE
    trtmc_model_ast
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_ast_fbank PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME ast_fbank
    COMMAND test_ast_fbank
  )
endif()
