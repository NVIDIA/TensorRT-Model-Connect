# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_chronos_bolt SHARED
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_chronos_bolt
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_chronos_bolt SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_chronos_bolt PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_chronos_bolt PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_chronos_bolt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_chronos_bolt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_chronos_bolt_forecast_contract
    ../tests/cpp/test_forecast_contract.cpp
  )
  target_include_directories(test_chronos_bolt_forecast_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_chronos_bolt_forecast_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_chronos_bolt_forecast_contract PRIVATE
    trtmc_model_chronos_bolt
  )
  target_compile_options(test_chronos_bolt_forecast_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME chronos_bolt_forecast_contract COMMAND test_chronos_bolt_forecast_contract)
  set_tests_properties(chronos_bolt_forecast_contract PROPERTIES LABELS cpu)
endif()
