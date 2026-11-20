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
  add_executable(test_lerobot_act_task_contract
    ../tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_lerobot_act_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_lerobot_act_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_lerobot_act_task_contract PRIVATE trtmc_model_lerobot_act)
  target_compile_options(test_lerobot_act_task_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME lerobot_act_task_contract COMMAND test_lerobot_act_task_contract)
  set_tests_properties(lerobot_act_task_contract PROPERTIES LABELS cpu)

  # Public C and C++ SDK consumers for the family-owned E2E test.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_lerobot_act_sdk_${_language}
      ../tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_lerobot_act_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_lerobot_act_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_lerobot_act_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  # stb_image's float conversions call libm, which a C link does not add.
  target_link_libraries(test_lerobot_act_sdk_c PRIVATE m)
  add_dependencies(test_lerobot_act_task_contract
    test_lerobot_act_sdk_c test_lerobot_act_sdk_cpp
  )

  add_executable(lerobot_act_qualification
    ../tests/cpp/qualification.cpp
  )
  target_include_directories(lerobot_act_qualification SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(lerobot_act_qualification PRIVATE
    trtmc_c
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(lerobot_act_qualification PRIVATE -Wall -Wextra -Wpedantic)
endif()
