# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_chroma SHARED
  pipeline.cpp
  plugin.cpp
  sampling.cpp
  t5_tokenizer.cpp
)

# The sampler mirrors the reference's individually rounded float operations.
set_source_files_properties(sampling.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

target_include_directories(trtmc_model_chroma
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_chroma SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_chroma PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
)
target_compile_options(trtmc_model_chroma PRIVATE
  -Wall -Wextra -Wpedantic
)
set_target_properties(trtmc_model_chroma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_chroma
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_chroma_host
    ${PROJECT_SOURCE_DIR}/families/chroma/tests/cpp/test_chroma_host.cpp
    sampling.cpp
    t5_tokenizer.cpp
  )
  target_include_directories(test_chroma_host PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(test_chroma_host PRIVATE
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_chroma_host PRIVATE
    -Wall -Wextra -Wpedantic
  )
  # The family E2E also runs it from TRTMC_NATIVE_BUILD_DIR for tokenizer parity.
  set_target_properties(test_chroma_host PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME chroma_host COMMAND test_chroma_host)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_chroma_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/chroma/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_chroma_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_chroma_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_chroma_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_chroma test_chroma_sdk_c test_chroma_sdk_cpp)
endif()
