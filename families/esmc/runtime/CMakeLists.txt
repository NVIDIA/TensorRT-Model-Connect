# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_esmc SHARED
  esmc_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_esmc
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_esmc SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_esmc PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_esmc PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_esmc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_esmc
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_esmc_tokenizer
    ${PROJECT_SOURCE_DIR}/families/esmc/tests/cpp/test_tokenizer.cpp
  )
  target_include_directories(test_esmc_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_esmc_tokenizer PRIVATE
    trtmc_model_esmc
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_esmc_tokenizer PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME esmc_tokenizer
    COMMAND test_esmc_tokenizer
  )

  add_executable(test_esmc_pipeline
    ${PROJECT_SOURCE_DIR}/families/esmc/tests/cpp/test_pipeline.cpp
  )
  target_include_directories(test_esmc_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_esmc_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_esmc_pipeline PRIVATE
    trtmc_model_esmc
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_esmc_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME esmc_pipeline
    COMMAND test_esmc_pipeline
  )
endif()
