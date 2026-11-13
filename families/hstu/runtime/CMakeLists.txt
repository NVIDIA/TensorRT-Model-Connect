# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hstu SHARED
  plugin.cpp
  record.cpp
)

target_include_directories(trtmc_model_hstu
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_hstu SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_hstu PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_hstu PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_hstu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_hstu
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_library(trtmc_cli_hstu SHARED cli.cpp)
target_include_directories(trtmc_cli_hstu PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_link_libraries(trtmc_cli_hstu PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_hstu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_hstu
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_hstu_record
    ${PROJECT_SOURCE_DIR}/families/hstu/tests/cpp/test_hstu_record.cpp
    ${PROJECT_SOURCE_DIR}/families/hstu/runtime/record.cpp
  )
  target_include_directories(test_hstu_record PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_hstu_record PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_hstu_record PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME hstu_record COMMAND test_hstu_record)
  set_tests_properties(hstu_record PROPERTIES LABELS "cpu")

  add_executable(hstu_task_probe ${PROJECT_SOURCE_DIR}/families/hstu/tests/task_probe.cpp)
  target_include_directories(hstu_task_probe PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(hstu_task_probe PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
  set_target_properties(hstu_task_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN"
  )
  add_dependencies(hstu_task_probe trtmc_backend_trt trtmc_model_hstu trtmc_cli_hstu)
  # Selective CI builds only targets behind add_test; the E2E needs the probe.
  add_dependencies(test_hstu_record hstu_task_probe)
endif()
