# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_cosyvoice3 SHARED
  bpe_tokenizer.cpp
  reference.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_cosyvoice3 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_cosyvoice3 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_cosyvoice3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_cosyvoice3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_cosyvoice3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_cosyvoice3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_cosyvoice3_runtime
    ${PROJECT_SOURCE_DIR}/families/cosyvoice3/tests/cpp/test_cosyvoice3_runtime.cpp
  )
  target_include_directories(test_cosyvoice3_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_cosyvoice3_runtime SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_cosyvoice3_runtime PRIVATE
    trtmc_model_cosyvoice3
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_cosyvoice3_runtime PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_cosyvoice3_runtime COMMAND test_cosyvoice3_runtime)
endif()
