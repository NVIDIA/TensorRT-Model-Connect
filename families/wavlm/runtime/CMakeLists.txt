# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wavlm SHARED
  ctc.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_wavlm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_wavlm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_wavlm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_wavlm PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_wavlm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_wavlm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_wavlm_ctc
    ${PROJECT_SOURCE_DIR}/families/wavlm/tests/cpp/test_ctc.cpp
    ${PROJECT_SOURCE_DIR}/families/wavlm/tests/cpp/test_pipeline.cpp
  )
  target_include_directories(test_wavlm_ctc PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_wavlm_ctc PRIVATE
    trtmc_model_wavlm
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_wavlm_ctc PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME wavlm_ctc
    COMMAND test_wavlm_ctc
  )
endif()
