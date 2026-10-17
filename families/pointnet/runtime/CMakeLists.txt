# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_pointnet SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_pointnet PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_pointnet SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_pointnet PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_pointnet PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_pointnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_pointnet
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_pointnet_task_contract
    ${PROJECT_SOURCE_DIR}/families/pointnet/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_pointnet_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_pointnet_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_pointnet_task_contract PRIVATE
    trtmc_model_pointnet trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_pointnet_task_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME pointnet_task_contract COMMAND test_pointnet_task_contract)
  set_tests_properties(pointnet_task_contract PROPERTIES LABELS "cpu")

  add_executable(pointnet_qualification
    ${PROJECT_SOURCE_DIR}/families/pointnet/tests/cpp/pointnet_qualification.cpp
  )
  target_include_directories(pointnet_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}/core/api/include
  )
  target_link_libraries(pointnet_qualification PRIVATE trtmc_c)
  target_compile_options(pointnet_qualification PRIVATE -Wall -Wextra -Wpedantic)
endif()
