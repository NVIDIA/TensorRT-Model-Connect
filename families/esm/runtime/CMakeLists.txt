# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_esm SHARED
  esm_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_esm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_esm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_esm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_esm PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_esm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_esm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_esm_tokenizer
    ${PROJECT_SOURCE_DIR}/families/esm/tests/cpp/test_tokenizer.cpp
  )
  target_include_directories(test_esm_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_esm_tokenizer PRIVATE
    trtmc_model_esm
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_esm_tokenizer PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME esm_tokenizer
    COMMAND test_esm_tokenizer
  )
endif()
