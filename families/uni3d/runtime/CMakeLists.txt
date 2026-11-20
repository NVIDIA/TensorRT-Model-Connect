# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_uni3d SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_uni3d PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_uni3d SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_uni3d PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_uni3d PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_uni3d PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_uni3d
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_uni3d_task_contract
    ${PROJECT_SOURCE_DIR}/families/uni3d/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_uni3d_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_uni3d_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_uni3d_task_contract PRIVATE
    trtmc_model_uni3d trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_uni3d_task_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME uni3d_task_contract COMMAND test_uni3d_task_contract)
  set_tests_properties(uni3d_task_contract PROPERTIES LABELS "cpu")

  add_executable(uni3d_qualification
    ${PROJECT_SOURCE_DIR}/families/uni3d/tests/cpp/uni3d_qualification.cpp
  )
  target_include_directories(uni3d_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}/core/api/include
  )
  target_link_libraries(uni3d_qualification PRIVATE trtmc_c)
  add_dependencies(uni3d_qualification trtmc_backend_trt trtmc_model_uni3d)
  target_compile_options(uni3d_qualification PRIVATE -Wall -Wextra -Wpedantic)
  # Selective CI builds only targets behind add_test; the E2E needs the qualifier.
  add_dependencies(test_uni3d_task_contract uni3d_qualification)
endif()
