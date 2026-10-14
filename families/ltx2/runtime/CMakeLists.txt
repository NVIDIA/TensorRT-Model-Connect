# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_ltx2 SHARED
  bpe_tokenizer.cpp
  distributed_runtime.cpp
  pipeline.cpp
  plugin.cpp
  runtime_config.cpp
)

target_include_directories(trtmc_model_ltx2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_ltx2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_ltx2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_ltx2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_ltx2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_ltx2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_ltx2_runtime_contract
    ../tests/cpp/test_runtime_contract.cpp
  )
  target_include_directories(test_ltx2_runtime_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_ltx2_runtime_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME ltx2_runtime_contract
    COMMAND test_ltx2_runtime_contract
  )
endif()
