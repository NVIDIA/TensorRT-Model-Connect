# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hubert SHARED
  ctc.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_hubert
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_hubert SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_hubert PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_hubert PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_hubert PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_hubert
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_hubert_ctc
    ${PROJECT_SOURCE_DIR}/families/hubert/tests/cpp/test_ctc.cpp
    ${PROJECT_SOURCE_DIR}/families/hubert/tests/cpp/test_pipeline.cpp
  )
  target_include_directories(test_hubert_ctc PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_hubert_ctc PRIVATE
    trtmc_model_hubert
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_hubert_ctc PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME hubert_ctc
    COMMAND test_hubert_ctc
  )

  add_executable(test_hubert_sdk_cpp ${PROJECT_SOURCE_DIR}/families/hubert/tests/sdk_consumer.cpp)
  target_link_libraries(test_hubert_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_hubert_sdk_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_hubert_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
