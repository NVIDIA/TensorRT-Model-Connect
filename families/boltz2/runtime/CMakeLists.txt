# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_boltz2 SHARED
  feature_bundle.cpp
  pipeline.cpp
  plugin.cpp
  prepared_request.cpp
  random_samples.cpp
)
target_include_directories(trtmc_model_boltz2 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_boltz2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_boltz2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_boltz2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_boltz2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_boltz2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_boltz2_sections
    ${PROJECT_SOURCE_DIR}/families/boltz2/tests/cpp/test_boltz2_sections.cpp
  )
  target_include_directories(test_boltz2_sections PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_boltz2_sections PRIVATE trtmc_model_boltz2)
  target_compile_options(test_boltz2_sections PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME boltz2_sections COMMAND test_boltz2_sections)
  set_tests_properties(boltz2_sections PROPERTIES LABELS cpu)
endif()
