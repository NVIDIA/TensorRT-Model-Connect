# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lerobot_act SHARED
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_lerobot_act
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lerobot_act SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lerobot_act PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lerobot_act PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_lerobot_act PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lerobot_act
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_lerobot_act_pipeline
    ../tests/cpp/test_pipeline.cpp
  )
  target_include_directories(test_lerobot_act_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_lerobot_act_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_lerobot_act_pipeline PRIVATE trtmc_model_lerobot_act)
  target_compile_options(test_lerobot_act_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME lerobot_act_pipeline COMMAND test_lerobot_act_pipeline)
  set_tests_properties(lerobot_act_pipeline PROPERTIES LABELS cpu)

  add_executable(lerobot_act_qualification
    ../tests/cpp/qualification.cpp
  )
  target_include_directories(lerobot_act_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(lerobot_act_qualification SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(lerobot_act_qualification PRIVATE
    trtmc_runtime
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(lerobot_act_qualification PRIVATE -Wall -Wextra -Wpedantic)
endif()
