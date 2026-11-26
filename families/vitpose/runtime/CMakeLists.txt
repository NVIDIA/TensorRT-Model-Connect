# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The image operations reproduce the reference processor's float32 and float64 rounding step by
# step; a fused multiply-add would round once where it rounds twice.
set(_vitpose_compile_options -Wall -Wextra -Wpedantic -ffp-contract=off)

add_library(trtmc_model_vitpose SHARED
  image_ops.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_vitpose PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_vitpose SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_vitpose PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_vitpose PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:${_vitpose_compile_options}>"
)
set_target_properties(trtmc_model_vitpose PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_vitpose
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The library carries code adapted from Transformers (Apache-2.0) and scipy (BSD-3-Clause); ship
# their notices with it.
install(FILES ${PROJECT_SOURCE_DIR}/families/vitpose/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/vitpose
)

if(TRTMC_BUILD_TESTS)
  # The image operations and the pipeline are compiled into the tests directly: the model library
  # exports only its factory.
  add_executable(test_vitpose_task_contract
    ${PROJECT_SOURCE_DIR}/families/vitpose/tests/cpp/test_task_contract.cpp
    image_ops.cpp
    pipeline.cpp
  )
  add_executable(vitpose_image_ops_tool
    ${PROJECT_SOURCE_DIR}/families/vitpose/tests/cpp/image_ops_tool.cpp
    image_ops.cpp
  )
  foreach(_target IN ITEMS test_vitpose_task_contract vitpose_image_ops_tool)
    target_include_directories(${_target} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(${_target} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(${_target} PRIVATE trtmc_core ${TRTMC_CUDART_LIBRARY})
    target_compile_options(${_target} PRIVATE ${_vitpose_compile_options} -Werror)
  endforeach()
  target_link_libraries(test_vitpose_task_contract PRIVATE nlohmann_json::nlohmann_json)
  set_target_properties(vitpose_image_ops_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME vitpose_task_contract COMMAND test_vitpose_task_contract)
  set_tests_properties(vitpose_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_vitpose_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/vitpose/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_vitpose_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_vitpose_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_vitpose_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  # Premerge builds the family's CTest targets by name; the Python parity test runs the image-ops
  # tool and the E2E runs the SDK consumers.
  add_dependencies(test_vitpose_task_contract
    vitpose_image_ops_tool test_vitpose_sdk_c test_vitpose_sdk_cpp
  )
endif()
