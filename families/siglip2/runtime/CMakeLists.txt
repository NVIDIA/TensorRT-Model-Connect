# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_siglip2 SHARED
  bpe_tokenizer.cpp
  naflex_preprocess.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_siglip2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_siglip2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_siglip2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_siglip2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_siglip2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_siglip2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(_test naflex_preprocess pipeline tokenizer)
    add_executable(test_siglip2_${_test}
      ${PROJECT_SOURCE_DIR}/families/siglip2/tests/cpp/test_${_test}.cpp
    )
    target_include_directories(test_siglip2_${_test} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_siglip2_${_test} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_siglip2_${_test} PRIVATE
      trtmc_model_siglip2
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    target_compile_options(test_siglip2_${_test} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME test_siglip2_${_test} COMMAND test_siglip2_${_test})
  endforeach()

  # Host preprocessing of one raw RGB image, written out for the exact parity check
  # against the reference processor in tests/test_e2e.py, which builds it by name.
  add_executable(siglip2_preprocess_dump
    ${PROJECT_SOURCE_DIR}/families/siglip2/tests/cpp/preprocess_dump.cpp
  )
  target_include_directories(siglip2_preprocess_dump PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(siglip2_preprocess_dump PRIVATE trtmc_model_siglip2)
  target_compile_options(siglip2_preprocess_dump PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(siglip2_preprocess_dump PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
