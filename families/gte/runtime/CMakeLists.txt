# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gte SHARED
  wordpiece_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_gte
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_gte SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_gte PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_gte PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_gte PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_gte
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/gte
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_gte_tokenizer
    ${PROJECT_SOURCE_DIR}/families/gte/tests/cpp/test_tokenizer.cpp
  )
  target_include_directories(test_gte_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_gte_tokenizer PRIVATE
    trtmc_model_gte
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_gte_tokenizer PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME gte_tokenizer
    COMMAND test_gte_tokenizer
  )
endif()
