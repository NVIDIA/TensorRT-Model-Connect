# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wav2vec2 SHARED
  ctc.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_wav2vec2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_wav2vec2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_wav2vec2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_wav2vec2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_wav2vec2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_wav2vec2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_wav2vec2_ctc
    ${PROJECT_SOURCE_DIR}/families/wav2vec2/tests/cpp/test_ctc.cpp
  )
  target_include_directories(test_wav2vec2_ctc PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_wav2vec2_ctc PRIVATE
    trtmc_model_wav2vec2
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_wav2vec2_ctc PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME wav2vec2_ctc
    COMMAND test_wav2vec2_ctc
  )
endif()
