# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_patchtsmixer SHARED
  plugin.cpp
  pipeline.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_patchtsmixer
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_patchtsmixer SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_patchtsmixer PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_patchtsmixer PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_patchtsmixer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS trtmc_model_patchtsmixer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_patchtsmixer_forecast_contract
    ../tests/cpp/test_forecast_contract.cpp
  )
  target_include_directories(test_patchtsmixer_forecast_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_patchtsmixer_forecast_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_patchtsmixer_forecast_contract PRIVATE
    trtmc_model_patchtsmixer
  )
  target_compile_options(test_patchtsmixer_forecast_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME patchtsmixer_forecast_contract COMMAND test_patchtsmixer_forecast_contract)
  set_tests_properties(patchtsmixer_forecast_contract PROPERTIES LABELS cpu)
endif()
