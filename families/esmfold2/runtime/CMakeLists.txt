# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Host code without CUDA or TensorRT: request parsing, featurization, sampling arithmetic, structure writers.
add_library(trtmc_esmfold2_host STATIC
  bundle_config.cpp
  contract.cpp
  fold.cpp
  protein.cpp
  random.cpp
  sampler.cpp
  structure_io.cpp
)
set_target_properties(trtmc_esmfold2_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(trtmc_esmfold2_host PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_esmfold2_host PRIVATE nlohmann_json::nlohmann_json)
find_package(Threads REQUIRED)
target_link_libraries(trtmc_esmfold2_host PUBLIC Threads::Threads)
target_compile_options(trtmc_esmfold2_host PRIVATE -Wall -Wextra -Wpedantic)

add_library(trtmc_model_esmfold2 SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_esmfold2 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_esmfold2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_esmfold2 PRIVATE
  trtmc_esmfold2_host
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_esmfold2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_esmfold2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_esmfold2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${PROJECT_SOURCE_DIR}/families/esmfold2/NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/esmfold2)

if(TRTMC_BUILD_TESTS)
  add_executable(test_esmfold2_host
    ${PROJECT_SOURCE_DIR}/families/esmfold2/tests/cpp/test_esmfold2_host.cpp
  )
  target_link_libraries(test_esmfold2_host PRIVATE trtmc_esmfold2_host nlohmann_json::nlohmann_json)
  target_compile_options(test_esmfold2_host PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(test_esmfold2_host PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME esmfold2_host COMMAND test_esmfold2_host)
  set_tests_properties(esmfold2_host PROPERTIES LABELS cpu)
endif()
