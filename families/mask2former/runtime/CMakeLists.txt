# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_mask2former SHARED
  image_ops.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_mask2former PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_mask2former SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_mask2former PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_mask2former PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_mask2former PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_mask2former
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The library carries code adapted from Pillow (MIT-CMU); ship its notice with it.
install(FILES ${PROJECT_SOURCE_DIR}/families/mask2former/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/mask2former
)

if(TRTMC_BUILD_TESTS)
  # The image operations are compiled into the tests directly: the model library exports only its factory.
  add_executable(test_mask2former_task_contract
    ${PROJECT_SOURCE_DIR}/families/mask2former/tests/cpp/test_task_contract.cpp
    image_ops.cpp
    pipeline.cpp
  )
  add_executable(mask2former_image_ops_tool
    ${PROJECT_SOURCE_DIR}/families/mask2former/tests/cpp/image_ops_tool.cpp
    image_ops.cpp
  )
  foreach(_target IN ITEMS test_mask2former_task_contract mask2former_image_ops_tool)
    target_include_directories(${_target} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(${_target} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(${_target} PRIVATE trtmc_core ${TRTMC_CUDART_LIBRARY})
    target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
  target_link_libraries(test_mask2former_task_contract PRIVATE nlohmann_json::nlohmann_json)
  set_target_properties(mask2former_image_ops_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME mask2former_task_contract COMMAND test_mask2former_task_contract)
  set_tests_properties(mask2former_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_mask2former_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/mask2former/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_mask2former_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_mask2former_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_mask2former_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  # Premerge builds the family's CTest targets by name; the Python parity test runs the image-ops tool.
  add_dependencies(test_mask2former_task_contract
    mask2former_image_ops_tool test_mask2former_sdk_c test_mask2former_sdk_cpp
  )
endif()
