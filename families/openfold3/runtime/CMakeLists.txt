# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_openfold3 SHARED
  feature_bundle.cpp
  pipeline.cpp
  plugin.cpp
  random_samples.cpp
)
target_include_directories(trtmc_model_openfold3 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_openfold3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_openfold3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_openfold3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_openfold3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_openfold3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_openfold3_sections
    ../tests/cpp/test_sections.cpp
  )
  target_include_directories(test_openfold3_sections PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_openfold3_sections PRIVATE trtmc_model_openfold3)
  target_compile_options(test_openfold3_sections PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME openfold3_sections COMMAND test_openfold3_sections)
  set_tests_properties(openfold3_sections PROPERTIES LABELS cpu)

  add_executable(openfold3_qualification
    ../tests/cpp/qualification.cpp
  )
  target_include_directories(openfold3_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(openfold3_qualification PRIVATE
    trtmc_runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_options(openfold3_qualification PRIVATE -Wall -Wextra -Wpedantic)
endif()
