# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_oneformer SHARED
  image_ops.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_oneformer
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_oneformer SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_oneformer PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_oneformer PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_oneformer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_oneformer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # CPU contract of the Task binding, input validation and post-processing,
  # driven through a fake engine.
  add_executable(test_oneformer_task_contract
    ${PROJECT_SOURCE_DIR}/families/oneformer/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_oneformer_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_oneformer_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_oneformer_task_contract PRIVATE
    trtmc_model_oneformer
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_oneformer_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  set_target_properties(test_oneformer_task_contract PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME oneformer_task_contract COMMAND test_oneformer_task_contract)
  set_tests_properties(oneformer_task_contract PROPERTIES LABELS "cpu")

  # Native preprocessing and score resize, exposed for the family's parity
  # test against torchvision.
  add_executable(test_oneformer_image_ops
    ${PROJECT_SOURCE_DIR}/families/oneformer/tests/cpp/image_ops_tool.cpp
  )
  target_include_directories(test_oneformer_image_ops PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_oneformer_image_ops SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_oneformer_image_ops PRIVATE
    trtmc_model_oneformer
  )
  target_compile_options(test_oneformer_image_ops PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  set_target_properties(test_oneformer_image_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  # Public SDK consumers for the single-device E2E; they link only the public C
  # runtime, and building the CPU contract builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_oneformer_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/oneformer/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_oneformer_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_oneformer_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_oneformer_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_oneformer_task_contract
    test_oneformer_image_ops
    test_oneformer_sdk_c
    test_oneformer_sdk_cpp
  )
endif()
