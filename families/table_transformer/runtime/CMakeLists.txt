# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_table_transformer SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_table_transformer
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_table_transformer SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_table_transformer PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_table_transformer PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_table_transformer PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_table_transformer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_table_transformer_preprocess_consumer
    ${PROJECT_SOURCE_DIR}/families/table_transformer/tests/preprocess_consumer.cpp
    image_preprocess_seam.cpp
  )
  target_include_directories(test_table_transformer_preprocess_consumer PRIVATE ${PROJECT_SOURCE_DIR})
  set_target_properties(test_table_transformer_preprocess_consumer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_executable(test_table_transformer_task_contract
    ${PROJECT_SOURCE_DIR}/families/table_transformer/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_table_transformer_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_table_transformer_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_table_transformer_task_contract PRIVATE
    trtmc_model_table_transformer
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_table_transformer_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME table_transformer_task_contract COMMAND test_table_transformer_task_contract)
  set_tests_properties(table_transformer_task_contract PROPERTIES LABELS "cpu")

  # Public SDK consumers for the selected single-device E2E. They link only the
  # public C runtime, and building the CPU contract builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_table_transformer_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/table_transformer/tests/sdk_consumer.${_language}
    )
    add_dependencies(test_table_transformer_sdk_${_language} test_table_transformer_preprocess_consumer)
    target_link_libraries(test_table_transformer_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_table_transformer_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_table_transformer_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_table_transformer_task_contract
    test_table_transformer_sdk_c
    test_table_transformer_sdk_cpp
  )

  add_executable(test_table_transformer_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/table_transformer/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_table_transformer_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_table_transformer_image_preprocess PRIVATE
    trtmc_model_table_transformer
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_table_transformer_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME table_transformer_image_preprocess
    COMMAND test_table_transformer_image_preprocess
  )
endif()
